# The test data of SAVI (Sheffield Accelerated Value of Information, BSD 3-clause):
# 1000 draws of a two-option model with 19 parameters, copied unchanged with its
# licence into test/fixtures/savi/. The reference values are written by
# validation/savi_reference.R with mgcv, fitting the regressions that SAVI fits.

function savi_columns(file)
    lines = readlines(joinpath(@__DIR__, "fixtures", "savi", file))
    rows = [parse.(Float64, split(line, ',')) for line in lines[2:end]]
    return Symbol.(split(lines[1], ',')), rows
end

@testset "SAVI test data" begin
    ref = JSON.parsefile(joinpath(@__DIR__, "fixtures", "savi", "savi_reference.json"))
    _, costs = savi_columns("costs.csv")
    _, effects = savi_columns("effects.csv")
    names, rows = savi_columns("parameters.csv")
    @test length(rows) == ref["draws"] == 1000
    @test length(names) == 19
    Q = [ref["lambda"] * effects[i][a] - costs[i][a] for a in 1:2, i in eachindex(costs)]
    draws = NamedTuple{Tuple(names)}(Tuple([row[j] for row in rows]
                                           for j in eachindex(names)))
    problem = DecisionProblem(PayoffTable(Q; actions=[:option1, :option2]),
                              SampleBelief(draws); baseline=:option1)

    @test [e.value for e in expected_utility_estimates(problem)] ≈
          ref["expected_net_benefit"] rtol = 1e-12
    @test evpi(problem, SampleAverage()).value ≈ ref["evpi"] rtol = 1e-12
    best = probability_optimal(problem)
    @test [b.probability for b in best] == [459 // 1000, 541 // 1000]
    @test Float64.([b.probability for b in best]) ≈ ref["probability_best"]
    @test sum(b.probability for b in best) == 1
    @test best[1].standard_error ≈ sqrt(0.459 * 0.541 / 1000) rtol = 1e-12

    # Two regressions of the same quantity on 1000 draws: they agree within the
    # Monte Carlo error of ours, and where one finds no dependence so does the other.
    for (name, target) in ref["evppi_single"]
        estimate = evppi(problem, Symbol(name))
        @test abs(estimate.value - target) <= 2 * estimate.standard_error + 1e-6
        @test isempty(estimate.details.redundant_parameters)
    end
    for name in ("theta6", "theta15", "theta16")
        @test evppi(problem, Symbol(name)).value ≈ ref["evppi_single"][name] rtol = 0.02
    end
    for group in ref["evppi_groups"]
        estimate = evppi(problem, Tuple(Symbol.(group["parameters"])))
        @test abs(estimate.value - group["evppi"]) <= estimate.standard_error
    end
end

@testset "redundant focal parameters" begin
    rng = Xoshiro(1)
    n = 2000
    a = randn(rng, n)
    b = randn(rng, n)
    Q = vcat(zeros(n)', (a .+ 0.5 .* b .+ 0.3 .* randn(rng, n))')
    problem(draws) = DecisionProblem(PayoffTable(Q; actions=[:none, :act]),
                                     SampleBelief(draws); baseline=:none)
    alone = evppi(problem((; a, b)), :a)
    @test isempty(alone.details.redundant_parameters)
    constant = evppi(problem((; a, c=fill(2.0, n), b)), (:a, :c))
    copy = evppi(problem((; a, d=2 .* a .+ 1, b)), (:a, :d))
    @test constant.value == alone.value == copy.value
    @test constant.details.redundant_parameters == [:c]
    @test copy.details.redundant_parameters == [:d]
    both = evppi(problem((; a, b)), (:a, :b))
    sum_too = evppi(problem((; a, b, s=a .+ b)), (:a, :b, :s))
    @test sum_too.value == both.value
    @test sum_too.details.redundant_parameters == [:s]
    # The first of two copies is the one kept, and a column is always kept.
    @test evppi(problem((; d=2 .* a .+ 1, a, b)), (:d, :a)).details.redundant_parameters ==
          [:a]
    flat = evppi(problem((; c=fill(2.0, n), a)), :c)
    @test flat.value ≈ 0 atol = 1e-12
    @test isempty(flat.details.redundant_parameters)
    # A method that is not a regression is given the columns as they are.
    @test isempty(evppi(problem((; a, b)), :a, StrongOakley()).details.redundant_parameters)
    # Non-numeric columns are never screened.
    g = rand(rng, ["x", "y"], n)
    @test isempty(evppi(problem((; a, g)), (:a, :g),
                        SplineRegression()).details.redundant_parameters)
end

@testset "probability that an action is best" begin
    table = PayoffTable([0 0 0; 4 -1 -1; 4 -2 0]; actions=[:none, :act, :other])
    exact = probability_optimal(DecisionProblem(table,
                                                FiniteBelief([1 // 2, 1 // 4, 1 // 4])))
    @test [r.probability for r in exact] == [1 // 2, 1 // 2, 3 // 4]   # ties count twice
    @test all(r -> r.standard_error === nothing, exact)
    limited = DecisionProblem(table, FiniteBelief([1 // 2, 1 // 4, 1 // 4]);
                              feasible=[:none, :act])
    @test [r.action for r in probability_optimal(limited)] == [:none, :act]
    @test [r.probability for r in probability_optimal(limited)] == [1 // 2, 1 // 2]
    # The act-now action is surely best exactly when perfect information is worth nothing.
    sure = DecisionProblem(PayoffTable([1 2; 0 2]), FiniteBelief([1 // 3, 2 // 3]))
    @test probability_optimal(sure)[1].probability == 1
    @test evpi(sure) == 0
    @test evpi(limited) > 0
    @test all(r -> r.probability < 1, probability_optimal(limited))
    @test_throws BeliefFormError probability_optimal(DecisionProblem(PayoffTable([1 2; 0 2]),
                                                                     CredalSet([[1 // 2,
                                                                                 1 // 2],
                                                                                [1 // 4,
                                                                                 3 // 4]])))
end
