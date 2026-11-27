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

    # The same regression as the reference (GAM.jl for mgcv, differences from the
    # first option, smoothing by GCV): the same estimate, not only a compatible one.
    for (name, target) in ref["evppi_single"]
        @test evppi(problem, Symbol(name), GAMRegression()).value ≈ target rtol = 1e-3 atol = 0.02
    end
    pairs_gam = [evppi(problem, Tuple(Symbol.(group["parameters"])),
                       GAMRegression(; tensor=true)).value
                 for group in ref["evppi_groups"]]
    targets = [group["evppi"] for group in ref["evppi_groups"]]
    @test pairs_gam[1] ≈ targets[1] rtol = 1e-4
    @test pairs_gam[2] ≈ targets[2] rtol = 1e-4
    # For theta7 with theta16 the GCV score has two local minima. GAM.jl's simplex
    # search ends in the worse one (score 0.017% higher; smoothing parameters 90 and
    # 5.1 against mgcv's 2e8 and 0.32) and the estimate is 0.7% higher. GAM.jl computes
    # the same score as mgcv at both points: only its search differs.
    @test pairs_gam[3] ≈ targets[3] rtol = 0.01
    @test_broken pairs_gam[3] ≈ targets[3] rtol = 1e-3
    # Smoothing by REML, against mgcv run to a tight tolerance (the score is flat in
    # the smoothing parameter and the estimate is not; see validation/README.md).
    for (name, target) in ref["evppi_single_reml"]
        @test evppi(problem, Symbol(name), GAMRegression(; smoothing=:REML)).value ≈
              target rtol = 2e-3 atol = 0.02
    end
    pairs_reml = [evppi(problem, Tuple(Symbol.(group["parameters"])),
                        GAMRegression(; tensor=true, smoothing=:REML)).value
                  for group in ref["evppi_groups"]]
    targets_reml = [group["evppi_reml"] for group in ref["evppi_groups"]]
    @test pairs_reml[1] ≈ targets_reml[1] rtol = 1e-3
    @test pairs_reml[3] ≈ targets_reml[3] rtol = 1e-3
    # For theta6 with theta15 GAM.jl ends in another basin of the REML score
    # (9825.7864 against mgcv's 9825.7469) and the estimate is 0.9% lower.
    @test pairs_reml[2] ≈ targets_reml[2] rtol = 0.01
    @test_broken pairs_reml[2] ≈ targets_reml[2] rtol = 1e-3
    @test_throws ArgumentError GAMRegression(; smoothing=:ML)
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

    # The same screen wherever a regression is fitted, not only in `evppi`: a constant
    # or duplicated focal parameter changes nothing.
    plain, wide = problem((; a, b)), problem((; a, c=fill(2.0, n), d=2 .* a .+ 1, b))
    spline = SplineRegression()
    @test DecisionAnalysis._fit_screened(spline, (; a, d=2 .* a .+ 1), Q,
                                         plain.belief.weights) ==
          DecisionAnalysis._fit(spline, (; a), Q, plain.belief.weights)
    quantity = a .^ 2 .+ b
    for method in
        (nothing, spline, GaussianProcessRegression(; max_points=300, hyper_points=100))
        @test expected_variance_reduction(wide.belief, (:a, :c, :d); quantity, method) ==
              expected_variance_reduction(plain.belief, :a; quantity, method)
        @test expected_variance_reduction(wide, (:a, :c, :d); quantity, method) ==
              expected_variance_reduction(plain, :a; quantity, method)
    end
    @test regression_check(wide, (:a, :c, :d), spline) ==
          regression_check(plain, :a, spline)
    @test compare_regressions(wide, (:a, :c, :d), [spline, GAMRegression()]) ==
          compare_regressions(plain, :a, [spline, GAMRegression()])
    # A method that is not a regression still gets the columns as they are.
    @test first(DecisionAnalysis._fit_columns(StrongOakley(), (; a, c=fill(2.0, n)))) ==
          (; a, c=fill(2.0, n))
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
