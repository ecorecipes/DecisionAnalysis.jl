# Aggregators against definitions written out by hand.

# Rockafellar-Uryasev: CVaR_α(X) = max_c { c − E[(c − X)⁺] / α }, the maximum attained at
# one of the values of X. Independent of the sort-and-accumulate implementation.
function oracle_cvar(x, p, α)
    return maximum(c - sum(p[i] * max(c - x[i], 0) for i in eachindex(x)) / α for c in x)
end

# Smallest value with cumulative probability at least α.
function oracle_quantile(x, p, α)
    return minimum(v for v in x if sum(p[i] for i in eachindex(x) if x[i] <= v) >= α)
end

@testset "aggregators" begin
    Q = [0 -10 -60 -150; -20 -20 -25 -90; -48 -48 -48 -48]
    t = PayoffTable([:nothing, :levee, :relocate], [:low, :moderate, :high, :extreme], Q)
    p = [1 // 2, 3 // 10, 3 // 20, 1 // 20]
    problem = DecisionProblem(t, p)

    @testset "worked values" begin
        @test aggregate(Expectation(), problem) == [-39 // 2, -97 // 4, -48]
        @test aggregate(WorstCase(), problem) == [-150, -90, -48]
        @test aggregate(BestCase(), problem) == [0, -20, -48]
        @test aggregate(Laplace(), problem) == [-55, -155 // 4, -48]
        @test aggregate(Hurwicz(1 // 4), problem) == [-225 // 2, -145 // 2, -48]
        @test aggregate(Hurwicz(0), problem) == aggregate(WorstCase(), problem)
        @test aggregate(Hurwicz(1), problem) == aggregate(BestCase(), problem)
        # CVaR at 20%: the worst 5% (extreme) and 15% (high) of the mass.
        @test aggregate(CVaR(1 // 5), problem)[1] == (-150 * 1 // 20 - 60 * 3 // 20) * 5
        @test aggregate(CVaR(1), problem) == aggregate(Expectation(), problem)
        @test aggregate(Quantile(1 // 20), problem) == [-150, -90, -48]
        @test aggregate(Quantile(1 // 5), problem) == [-60, -25, -48]
        @test aggregate(Quantile(1), problem) == [0, -20, -48]
    end

    @testset "parameter checks" begin
        @test_throws ArgumentError Hurwicz(1.5)
        @test_throws ArgumentError Hurwicz(-0.1)
        @test_throws ArgumentError AlphaMaxmin(2)
        @test_throws ArgumentError CVaR(0)
        @test_throws ArgumentError CVaR(1.1)
        @test_throws ArgumentError Quantile(0.0)
    end

    @testset "belief forms" begin
        scen = DecisionProblem(t, ScenarioSet([:low, :moderate]))
        @test aggregate(WorstCase(), scen) == [-10, -20, -48]
        @test aggregate(BestCase(), scen) == [0, -20, -48]
        @test aggregate(Laplace(), scen) == [-5, -20, -48]
        for agg in
            (Expectation(), LowerExpectation(), UpperExpectation(), AlphaMaxmin(0.5),
             CVaR(0.5), Quantile(0.5))
            @test_throws BeliefFormError aggregate(agg, scen)
        end
        point = DecisionProblem(t, PointBelief(:high))
        for agg in (Expectation(), WorstCase(), BestCase(), Laplace(), Hurwicz(1 // 4),
                    LowerExpectation(), UpperExpectation(), CVaR(0.5), Quantile(0.5))
            @test aggregate(agg, point) == [-60, -25, -48]
        end
        credal = DecisionProblem(t,
                                 CredalSet(t.states,
                                           [p, [7 // 20, 3 // 10, 1 // 4,
                                                1 // 10]]))
        @test_throws BeliefFormError aggregate(Expectation(), credal)
        @test_throws BeliefFormError aggregate(CVaR(0.5), credal)
        # State-wise aggregators see every state of the table under a credal set.
        @test aggregate(WorstCase(), credal) == [-150, -90, -48]
        # A single prior is a one-point credal set.
        @test aggregate(LowerExpectation(), problem) == aggregate(Expectation(), problem)
        @test aggregate(UpperExpectation(), problem) == aggregate(Expectation(), problem)
        @test aggregate(AlphaMaxmin(1 // 3), problem) == aggregate(Expectation(), problem)
        # A state of probability zero still counts for a state-wise aggregator, and not
        # for a probabilistic one.
        zero_extreme = DecisionProblem(t, [1 // 2, 3 // 10, 1 // 5, 0 // 1])
        @test aggregate(WorstCase(), zero_extreme)[1] == -150
        @test aggregate(Quantile(1 // 100), zero_extreme)[1] == -60
        @test aggregate(CVaR(1 // 100), zero_extreme)[1] == -60
    end

    @testset "random problems against oracles" begin
        rng = Random.Xoshiro(11)
        for _ in 1:40
            m, n = rand(rng, 1:4), rand(rng, 1:6)
            prob = exact_problem(rng, m, n)
            Q, q = prob.payoff.Q, probabilities(prob.belief)
            @test aggregate(Expectation(), prob) == oracle_expected_utility(Q, q)
            α = rand(rng, 1:20) // 20
            @test aggregate(CVaR(α), prob) == [oracle_cvar(Q[a, :], q, α) for a in 1:m]
            @test aggregate(Quantile(α), prob) ==
                  [oracle_quantile(Q[a, :], q, α) for a in 1:m]
            @test aggregate(Hurwicz(α), prob) ==
                  [α * maximum(Q[a, :]) + (1 - α) * minimum(Q[a, :]) for a in 1:m]
            # CVaR lies between the worst case and the mean and rises with α.
            cvar = aggregate(CVaR(α), prob)
            @test all(aggregate(WorstCase(), prob) .<= cvar .<=
                      aggregate(Expectation(), prob))
            @test all(aggregate(CVaR(α / 2), prob) .<= cvar)
            # The same in floating point.
            fprob = DecisionProblem(PayoffTable(Float64.(Q)), Float64.(q))
            @test aggregate(CVaR(Float64(α)), fprob) ≈ Float64.(cvar)
            @test aggregate(Expectation(), fprob) ≈ Float64.(aggregate(Expectation(), prob))
            # Prior-set aggregators against a loop over the points.
            cs = random_credal_set(rng, n, rand(rng, 1:4); T=BigRat)
            cprob = DecisionProblem(prob.payoff, cs)
            E = [oracle_expected_utility(Q, pt) for pt in extreme_points(cs)]
            lower = [minimum(e[a] for e in E) for a in 1:m]
            upper = [maximum(e[a] for e in E) for a in 1:m]
            @test aggregate(LowerExpectation(), cprob) == lower
            @test aggregate(UpperExpectation(), cprob) == upper
            @test aggregate(AlphaMaxmin(α), cprob) == α .* lower .+ (1 - α) .* upper
            @test aggregate(AlphaMaxmin(1), cprob) == lower
            @test aggregate(AlphaMaxmin(0), cprob) == upper
        end
    end
end
