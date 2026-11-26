# Sample-based value of information. Every estimator is checked two ways: against exact
# enumeration on a finite problem turned into a sample (the oracles of helpers.jl), and
# against the closed forms of the normal model of the R `voi` vignette.

# A finite problem over the product of two discrete parameters, as a sample whose draws
# are its states with the prior as weights: nothing is lost, so estimators that are
# exact for discrete parameters must reproduce the finite answer.
function product_sample(rng, m, k, l)
    n = k * l
    t = random_payoff_table(rng, m, n; T=BigRat)
    w = rand(rng, 1:5, n)
    g = repeat(1:k; inner=l)
    h = repeat(1:l; outer=k)
    belief = SampleBelief((; g, h, site=string.("site", g)); weights=w)
    return DecisionProblem(t, belief), t.Q, BigRat.(w) .// sum(w), g, h
end

@testset "GAM regression needs GAM.jl" begin
    @test Base.get_extension(DecisionAnalysis, :DecisionAnalysisGAMExt) === nothing
    p = voi_normal_sample(Xoshiro(1), 50)
    @test_throws MissingExtensionError evppi(p, :p1, GAMRegression())
    err = try
        evpi(p, GAMRegression())
    catch e
        e
    end
    @test err isa MissingExtensionError && occursin("using GAM", sprint(showerror, err))
end

using GAM: GAM

@testset "sample-based value of information" begin
    @testset "exact enumeration: discrete parameters" begin
        rng = Xoshiro(11)
        for _ in 1:25
            problem, Q, p, g, h = product_sample(rng, rand(rng, 2:4), rand(rng, 2:4),
                                                 rand(rng, 2:4))
            by_g = oracle_evsi(Q, p, partition_likelihood(g, BigRat))
            by_h = oracle_evsi(Q, p, partition_likelihood(h, BigRat))
            full = oracle_evpi(Q, p)

            # Grouping is exact, in exact arithmetic.
            e = evppi(problem, :g, Grouping())
            @test e.value == by_g && e.value isa Rational
            @test e.quantity === :evppi && e.method === :Grouping
            @test evppi(problem, :site).value == by_g          # non-numeric: Grouping
            @test evppi(problem, :h, Grouping()).value == by_h
            @test evppi(problem, (:g, :h), Grouping()).value == full
            grouped = conditional_problem(problem, :g)
            @test evpi(grouped) == by_g
            @test grouped.belief isa FiniteBelief
            @test aggregate(Expectation(), grouped) == oracle_expected_utility(Q, p)
            @test evppi(problem, [:g, :h], Grouping()).value == full

            # The built-in regression treats a column with few levels as a factor, and
            # reproduces the cell means; with interactions it is saturated.
            @test evppi(problem, :g, SplineRegression()).value ≈ Float64(by_g) atol = 1e-8
            @test evppi(problem, :site, SplineRegression()).value ≈ Float64(by_g) atol = 1e-8
            @test evppi(problem, (:g, :h), SplineRegression(; interactions=true)).value ≈
                  Float64(full) atol = 1e-8
            two = DecisionProblem(problem.payoff,
                                  SampleBelief((; g, h); weights=problem.belief.weights))
            @test evpi(two, SplineRegression(; interactions=true)).value ≈ Float64(full) atol = 1e-8

            # The plain estimator is the definition, with errors beside it.
            s = evpi(problem, SampleAverage())
            @test s.value == full == evpi(problem)
            @test s.standard_error >= 0 && s.bias.act_now >= 0 && s.bias.cells == 0.0
            @test s.details.act_now_value == maximum(oracle_expected_utility(Q, p))

            # No information, no value: a constant column.
            constant = DecisionProblem(problem.payoff,
                                       SampleBelief((; c=fill(1, length(p)));
                                                    weights=problem.belief.weights))
            @test iszero(evppi(constant, :c, Grouping()).value)
        end
    end

    @testset "exact enumeration: single-parameter methods" begin
        rng = Xoshiro(12)
        for _ in 1:25
            k, l, m = rand(rng, 3:6), rand(rng, 2:5), rand(rng, 2:4)
            n = k * l
            Q = BigRat.(rand(rng, -9:9, m, n))
            g = repeat(1:k; inner=l)
            p = fill(BigRat(1, n), n)
            exact = Float64(oracle_evsi(Q, p, partition_likelihood(g, BigRat)))
            order = Random.shuffle(rng, 1:n)     # the draws need not come sorted
            problem = DecisionProblem(PayoffTable(Q[:, order]),
                                      SampleBelief((; g=g[order])))
            # Strong and Oakley with one block per level is the exact conditional mean.
            @test evppi(problem, :g, StrongOakley(; blocks=k)).value ≈ exact atol = 1e-9
            @test evppi(problem, :g, Grouping(; bins=k)).value ≈ exact atol = 1e-9
            # One block is no information.
            @test evppi(problem, :g, StrongOakley(; blocks=1)).value ≈ 0 atol = 1e-12
            # A segmentation can never beat knowing the parameter exactly.
            @test evppi(problem, :g, Sadatsafavi()).value <= exact + 1e-9
            @test evppi(problem, :g, Sadatsafavi(; separators=2)).value <= exact + 1e-9
            @test evppi(problem, :g, Sadatsafavi(; separators=2)).value >=
                  evppi(problem, :g, Sadatsafavi()).value - 1e-9
        end
        # Sadatsafavi is exact when the best action changes once along the parameter
        # (one separator) or twice (two separators).
        for (separators, pattern) in
            ((1, [-3, -1, -1, 2, 4, 5]), (2, [3, 1, -2, -4, 1, 2]))
            l = 4
            g = repeat(1:6; inner=l)
            noise = repeat([-2, -1, 1, 2]; outer=6)
            Q = BigRat.(permutedims(hcat(zeros(Int, 24), pattern[g] .+ noise)))
            p = fill(BigRat(1, 24), 24)
            exact = Float64(oracle_evsi(Q, p, partition_likelihood(g, BigRat)))
            problem = DecisionProblem(PayoffTable(Q), SampleBelief((; g)))
            e = evppi(problem, :g, Sadatsafavi(; separators))
            @test e.value ≈ exact atol = 1e-9
            @test length(e.details.cut_points) == separators
        end
        p = voi_normal_sample(Xoshiro(1), 30)
        @test_throws ArgumentError evppi(p, (:p1, :p2), StrongOakley())
        @test_throws ArgumentError evppi(p, (:p1, :p2), Sadatsafavi())
        @test_throws ArgumentError evppi(p, :p3, StrongOakley())
        @test_throws ArgumentError evppi(p, :p1, SampleAverage())
        @test_throws ArgumentError evpi(p, StrongOakley())
        @test_throws ArgumentError Sadatsafavi(; separators=3)
        @test_throws ArgumentError StrongOakley(; blocks=0)
        @test_throws ArgumentError SplineRegression(; knots=-1)
        @test_throws ArgumentError evpi(p, SplineRegression(; knots=20, interactions=true))
        finite = DecisionProblem(PayoffTable([1 2; 3 0]), [1 // 2, 1 // 2])
        @test_throws BeliefFormError evpi(finite, SampleAverage())
        @test_throws BeliefFormError evppi(finite, :x, Grouping())
        @test_throws BeliefFormError evppi(finite, :x)
        @test_throws ArgumentError evpi(DecisionProblem(PayoffTable([1 2; 3 0]),
                                                        SampleBelief(2)),
                                        SplineRegression())
    end

    @testset "exact enumeration: EVSI" begin
        rng = Xoshiro(13)
        for _ in 1:25
            m, n, z = rand(rng, 2:4), rand(rng, 2:6), rand(rng, 2:4)
            t = random_payoff_table(rng, m, n; T=BigRat)
            w = rand(rng, 1:5, n)
            p = BigRat.(w) .// sum(w)
            L = random_stochastic(rng, BigRat, z, n)
            exact = oracle_evsi(t.Q, p, L)
            problem = DecisionProblem(t, SampleBelief(n; weights=w))
            # Importance sampling with every possible data set, each weighted by its
            # probability, is the exact preposterior calculation. A likelihood of zero
            # is a log-likelihood of −Inf; signals of probability zero are left out.
            marginal = [sum(p[j] * L[k, j] for j in 1:n) for k in 1:z]
            keep = findall(>(0), marginal)
            study = ImportanceSampling(log.(Float64.(L[keep, :]));
                                       weights=Float64.(marginal[keep]))
            e = evsi(problem, study)
            @test e.value ≈ Float64(exact) atol = 1e-9
            @test e.details.direct_value ≈ Float64(exact) atol = 1e-9
            @test e.quantity === :evsi && e.method === :ImportanceSampling
            @test 1 <= e.details.minimum_ess <= e.details.median_ess <= n + 1e-9
        end
        # A summary that is a function of the state: regression on it (as a factor) is
        # the exact value of the partition.
        for _ in 1:10
            problem, Q, p, g, h = product_sample(rng, 3, 3, 4)
            exact = Float64(oracle_evsi(Q, p, partition_likelihood(g, BigRat)))
            @test evsi(problem, SummaryStatistics((; label=string.(g)))).value ≈ exact atol = 1e-8
            # The Gaussian approximation runs from nothing to the EVPPI.
            curve = evsi(problem, GaussianApproximation(:g; n0=5, n=[0, 10, 1e12]))
            @test curve[1].value ≈ 0 atol = 1e-9
            @test curve[1].value <= curve[2].value + 1e-9 <= curve[3].value + 2e-9
            @test curve[3].value ≈ exact atol = 1e-4
            @test evsi(problem, GaussianApproximation(:g; n0=5, n=10)).value ==
                  curve[2].value
        end
        # A noisy signal: draws of (state, signal) pairs. The regression on the signal
        # estimates the exact EVSI of the finite experiment within its Monte Carlo error.
        Qf = Float64[0 0 0; 3 1 -4; -2 2 1]
        pf = [0.5, 0.3, 0.2]
        Lf = [0.7 0.2 0.1; 0.2 0.6 0.2; 0.1 0.2 0.7]
        exact = oracle_evsi(Qf, pf, Lf)
        n = 30_000
        θ = [searchsortedfirst(cumsum(pf), rand(rng)) for _ in 1:n]
        signal = [searchsortedfirst(cumsum(Lf[:, s]), rand(rng)) for s in θ]
        problem = DecisionProblem(PayoffTable(Qf[:, θ]), SampleBelief(n))
        e = evsi(problem, SummaryStatistics((; signal=string.(signal))))
        @test abs(e.value - exact) < 4 * e.standard_error + 0.01
        # The same by importance sampling, from the log-likelihood of each simulated
        # signal under each draw.
        K = 300
        loglik = [log(Lf[signal[k], θ[j]]) for k in 1:K, j in 1:n]
        i = evsi(problem, ImportanceSampling(loglik; weights=:equal))
        @test abs(i.value - exact) < 4 * i.standard_error + 0.01
        @test i.details.datasets == K

        @test_throws ArgumentError ImportanceSampling(zeros(0, 3))
        @test_throws ArgumentError ImportanceSampling([0.0 NaN])
        @test_throws ArgumentError ImportanceSampling([0.0 Inf])
        @test_throws ArgumentError ImportanceSampling(zeros(2, 3); weights=[1.0])
        small = DecisionProblem(PayoffTable([1 2 0; 0 1 3]), SampleBelief(3))
        @test_throws ArgumentError evsi(small, ImportanceSampling(zeros(2, 3)))
        @test_throws ArgumentError evsi(small, ImportanceSampling(zeros(3, 2)))
        @test_throws ArgumentError evsi(small, ImportanceSampling(fill(-Inf, 3, 3)))
        @test_throws ArgumentError evsi(small, SummaryStatistics((; t=[1.0, 2.0])))
        @test_throws ArgumentError SummaryStatistics((;))
        @test_throws ArgumentError GaussianApproximation(:g; n0=0, n=1)
        @test_throws ArgumentError GaussianApproximation(:g; n0=1, n=-1)
    end

    @testset "the R voi vignette normal model: closed forms" begin
        # Analytic EVPI 0.4798107 (the vignette's printed value); EVPPI and EVSI from
        # the same closed form E max(0, X) − max(0, E X).
        analytic_evpi = voi_normal_value(sqrt(5))
        @test analytic_evpi ≈ 0.4798107 atol = 5e-7
        evppi1, evppi2 = voi_normal_value(1.0), voi_normal_value(2.0)
        @test evppi1 ≈ 0.0833155 atol = 5e-7
        @test evppi2 ≈ 0.3955931 atol = 5e-7

        n = 40_000
        problem = voi_normal_sample(Xoshiro(20261123), n)
        within(e, truth; slack=0.0) = abs(e.value - truth) < 4 * e.standard_error + slack

        e = evpi(problem, SampleAverage())
        @test within(e, analytic_evpi)
        @test e.value == evpi(problem)
        @test 0.001 < e.standard_error < 0.01
        @test within(evpi(problem, SplineRegression()), analytic_evpi)
        @test within(evppi(problem, (:p1, :p2), SplineRegression()), analytic_evpi)

        for (focal, truth) in ((:p1, evppi1), (:p2, evppi2))
            @test within(evppi(problem, focal), truth)                # default: spline
            @test within(evppi(problem, focal, SplineRegression(; knots=0, degree=1)),
                         truth)
            @test within(evppi(problem, focal, StrongOakley()), truth; slack=0.005)
            @test within(evppi(problem, focal, Sadatsafavi()), truth; slack=0.005)
            @test within(evppi(problem, focal, Sadatsafavi(; separators=2)), truth;
                         slack=0.005)
            @test within(evppi(problem, focal, Grouping(; bins=25)), truth; slack=0.005)
        end
        @test evppi(problem, :p1).method === :SplineRegression

        # EVSI for p1 from m normal observations with known sd 1: prior effective
        # sample size 1, closed forms 0.0723251, 0.0821176, 0.0831946.
        truth(m) = voi_normal_value(sqrt(m / (m + 1)))
        @test truth(10) ≈ 0.0723251 atol = 5e-7
        @test truth(100) ≈ 0.0821176 atol = 5e-7
        @test truth(1000) ≈ 0.0831946 atol = 5e-7
        p1 = problem.belief.draws.p1
        rng = Xoshiro(5)
        for m in (10, 100)
            xbar = p1 .+ randn(rng, n) ./ sqrt(m)
            @test within(evsi(problem, SummaryStatistics((; xbar))), truth(m))
            ga = evsi(problem,
                      GaussianApproximation(:p1; n0=1, n=m,
                                            regression=SplineRegression(; knots=0,
                                                                        degree=1)))
            @test within(ga, truth(m))
            @test ga.method === :GaussianApproximation &&
                  ga.details.shrinkage ≈
                  sqrt(m / (m + 1))
        end
        curve = evsi(problem, GaussianApproximation(:p1; n0=1, n=[10, 100, 1000]))
        @test issorted([c.value for c in curve])
        @test all(within(c, truth(m); slack=0.003)
                  for (c, m) in zip(curve, (10, 100, 1000)))

        # Importance sampling on a smaller sample, with the helper that builds the
        # log-likelihood matrix from a data simulator and a log-likelihood.
        small = voi_normal_sample(Xoshiro(3), 4000)
        m = 10
        study = importance_loglikelihoods((rng, θ) -> θ.p1 + randn(rng) / sqrt(m),
                                          (x, θ) -> -m * (x - θ.p1)^2 / 2, small.belief;
                                          datasets=600, seed=9)
        is = evsi(small, study)
        @test within(is, truth(m); slack=0.01)
        @test is.details.median_ess > 50
        again = importance_loglikelihoods((rng, θ) -> θ.p1 + randn(rng) / sqrt(m),
                                          (x, θ) -> -m * (x - θ.p1)^2 / 2, small.belief;
                                          datasets=600, seed=9)
        @test again.loglikelihood == study.loglikelihood
        one_each = importance_loglikelihoods((rng, θ) -> θ.p1 + randn(rng) / sqrt(m),
                                             (x, θ) -> -m * (x - θ.p1)^2 / 2,
                                             SampleBelief((; p1=p1[1:200])); seed=9)
        @test size(one_each.loglikelihood) == (200, 200) && one_each.weights === nothing
        @test_throws ArgumentError importance_loglikelihoods((r, θ) -> 0.0, (x, θ) -> 0.0,
                                                             small.belief)
        @test_throws ArgumentError importance_loglikelihoods((r, θ) -> 0.0, (x, θ) -> 0.0,
                                                             SampleBelief(3); seed=1)

        # enbs: additive payoffs only, in the payoff's units.
        net = enbs(is; cost=0.01, population=3)
        @test net.value ≈ 3 * is.value - 0.01
        @test net.standard_error ≈ 3 * is.standard_error
        @test_throws ArgumentError enbs(e; cost=1)
        @test_throws ArgumentError enbs(is; cost=1, population=-1)

        # The bootstrap standard error covers the refitting and is of the same size.
        boot = evppi(small, :p1, SplineRegression(); bootstrap=60, seed=4)
        plain = evppi(small, :p1, SplineRegression())
        @test boot.value == plain.value
        @test 0.4 < boot.standard_error / plain.standard_error < 2.5
        @test boot.details.bootstrap == 60
        @test evppi(small, :p1, SplineRegression(); bootstrap=60, seed=4).standard_error ==
              boot.standard_error
        @test Float64(plain) == plain.value
        @test occursin("standard error", sprint(show, plain))

        # GAM.jl through the extension, on the same model.
        g = evppi(small, :p1, GAMRegression())
        @test g.method === :GAMRegression
        @test within(g, evppi1; slack=0.01)
        @test within(evpi(small, GAMRegression()), analytic_evpi; slack=0.02)
        @test within(evppi(small, (:p1, :p2), GAMRegression(; tensor=true)), analytic_evpi;
                     slack=0.03)
        @test_throws ArgumentError GAMRegression(; k=2)
    end

    @testset "noisy cells, the two biases and the variance split" begin
        # The normal model with simulation noise of standard error 1 in every cell of
        # the second action: the plain EVPI is inflated, the bias estimate says by how
        # much, and the all-parameter regression recovers the noise-free value.
        n = 20_000
        rng = Xoshiro(77)
        clean = voi_normal_sample(rng, n)
        se = vcat(zeros(1, n), ones(1, n))
        noisy_Q = clean.payoff.Q .+ se .* randn(rng, 2, n)
        noisy = DecisionProblem(PayoffTable(noisy_Q; actions=[:reference, :new],
                                            standard_errors=se,
                                            replicates=fill(4, 2, n)), clean.belief)
        truth = voi_normal_value(sqrt(5))
        plain = evpi(noisy, SampleAverage())
        inflated = voi_normal_value(sqrt(6)) - truth    # the noise adds 1 to the variance
        @test plain.value - truth > 0.5 * inflated
        @test plain.bias.cells ≈ plain.value - evpi(clean) rtol = 0.5
        @test plain.bias.cells > 0
        regression = evpi(noisy, SplineRegression())
        @test abs(regression.value - truth) < 4 * regression.standard_error + 0.005
        @test regression.bias.cells === nothing

        # Variance split against a slow oracle.
        parts = variance_components(noisy)
        w = Float64.(noisy.belief.weights)
        for (a, part) in enumerate(parts)
            mean_a = sum(w[i] * noisy_Q[a, i] for i in 1:n)
            total = sum(w[i] * (noisy_Q[a, i] - mean_a)^2 for i in 1:n)
            simulation = sum(w[i] * se[a, i]^2 for i in 1:n)
            @test part.total ≈ total && part.simulation ≈ simulation
            @test part.parameter ≈ total - simulation
        end
        @test parts[2].parameter ≈ 5 rtol = 0.1          # Var(p1 − p2) = 5
        @test parts[2].parameter_share ≈ 5 / 6 rtol = 0.05
        @test [part.action for part in parts] == [:reference, :new]
        @test_throws ArgumentError variance_components(clean)

        # The act-now bias against its closed form for two actions:
        # E max(X, Y) − max = sqrt(Var(X − Y) / 2π) when the means are equal.
        k = 400
        x = randn(rng, k)
        Q = permutedims(hcat(x .- sum(x) / k, zeros(k)))
        p = DecisionProblem(PayoffTable(Q), SampleBelief(k))
        expected = sqrt(sum(abs2, Q[1, :]) / k^2 / (2π))
        @test evpi(p, SampleAverage()).bias.act_now ≈ expected rtol = 0.1
        # One action: nothing to select, no bias.
        one = DecisionProblem(PayoffTable(Q[1:1, :]), SampleBelief(k))
        @test evpi(one, SampleAverage()).bias.act_now == 0
        # The cell-noise bias against the same closed form, in one cell.
        cell = DecisionProblem(PayoffTable([1.0; 1.0;;]; standard_errors=[0.3; 0.4;;]),
                               SampleBelief(1))
        @test evpi(cell, SampleAverage()).bias.cells ≈ sqrt((0.3^2 + 0.4^2) / (2π)) rtol = 0.2
        # Noise shared between the actions cancels: with correlation ρ the bias scales
        # by √(1 − ρ), and vanishes at ρ = 1.
        @test evpi(cell, SampleAverage(); noise_correlation=0.75).bias.cells ≈
              0.5 * sqrt((0.3^2 + 0.4^2) / (2π)) rtol = 0.2
        @test evpi(cell, SampleAverage(); noise_correlation=1).bias.cells == 0
        @test_throws ArgumentError evpi(cell, SampleAverage(); noise_correlation=2)
        # The achieved correlation of an outcome table, against explicit loops: runs
        # that share a replicate effect across actions, and runs that do not.
        shared = [(; draw=d, action=a, replicate=r,
                   n=10.0 * d + 3 * sin(7.0 * d + r) + 0.01 * cos(13.0 * r + (a == :b)))
                  for d in 1:5 for a in (:a, :b) for r in 1:6]
        @test crn_correlation(OutcomeTable(shared; outcomes=[:n]), IndividualsUtility(:n)) >
              0.99
        @test crn_correlation(OutcomeTable(shared; outcomes=[:n]), o -> o.n) > 0.99
        num, den = 0.0, 0.0
        for d in 1:5
            x = [row.n for row in shared if row.draw == d && row.action == :a]
            y = [row.n for row in shared if row.draw == d && row.action == :b]
            x = x .- sum(x) / 6
            y = y .- sum(y) / 6
            num += sum(x .* y)
            den += sqrt(sum(x .^ 2) * sum(y .^ 2))
        end
        @test crn_correlation(OutcomeTable(shared; outcomes=[:n]), IndividualsUtility(:n)) ≈
              num / den
        @test isnan(crn_correlation(OutcomeTable([(; draw=1, action=:a, replicate=1, n=1)];
                                                 outcomes=[:n]), IndividualsUtility(:n)))

        # Expected utilities with their two errors, against explicit loops.
        est = expected_utility_estimates(noisy)
        for (a, row) in enumerate(est)
            mean_a = sum(w[i] * noisy_Q[a, i] for i in 1:n)
            @test row.value ≈ mean_a
            @test row.standard_error ≈ sqrt(sum(w[i]^2 * (noisy_Q[a, i] - mean_a)^2
                                                for i in 1:n))
            @test row.simulation_error ≈ sqrt(sum(w[i]^2 * se[a, i]^2 for i in 1:n))
        end
        @test expected_utility_estimates(clean)[1].simulation_error === nothing
        based = DecisionProblem(noisy.payoff, noisy.belief; baseline=:reference)
        gains = expected_gain_estimates(based)
        @test [g.value for g in gains] ≈ expected_gain_over_baseline(based)
        @test gains[1].standard_error == 0
        @test gains[2].standard_error ≈ est[2].standard_error
        @test_throws MissingBaselineError expected_gain_estimates(noisy)
    end

    @testset "an estimate above the EVPI of its sample is flagged" begin
        # No intervention is best at every draw, so the EVPI of the sample is exactly
        # zero and nothing can be learned. The payoff of acting has a deep step that a
        # cubic spline overshoots beside it, so its fitted values rise above zero there:
        # the regression reports a positive EVPPI with a small standard error.
        n = 100
        x = collect(1:n) ./ n
        act = [0.4 < v <= 0.5 ? -40.0 : -1.0 for v in x]
        problem = DecisionProblem(PayoffTable([zeros(n)'; act']; actions=[:none, :act]),
                                  SampleBelief((; x)); baseline=:none)
        @test iszero(evpi(problem))
        e = @test_logs (:warn, r"above the EVPI of the same sample") evppi(problem, :x,
                                                                           SplineRegression())
        @test e.value > 0.5 && e.standard_error < 0.05
        @test e.details.above_sample_evpi
        @test iszero(e.details.sample_evpi)
        # Following the fitted policy loses on the payoffs themselves.
        @test e.details.policy_gain < 0
        @test occursin("ABOVE the EVPI of the sample", sprint(show, e))
        # The value is returned as computed, not capped.
        @test e.value == Float64(e)
        # The same through the default method and in the report's notes.
        @test_logs (:warn,) evppi(problem, :x)
        report = @test_logs (:warn,) match_mode = :any DecisionReport(problem;
                                                                      parameters=:x)
        @test any(occursin("above the EVPI of the same sample", note)
                  for note in report.assumptions)
        # Cell means cannot exceed it (the maximum of a mean is at most the mean of
        # the maxima), on this problem or on a noisy one, and are never flagged.
        for method in (StrongOakley(), Sadatsafavi(), Sadatsafavi(; separators=2),
                       Grouping(; bins=5))
            c = @test_logs evppi(problem, :x, method)
            @test iszero(c.value) && !c.details.above_sample_evpi
            @test iszero(c.details.policy_gain)
        end
        rng = Xoshiro(11)
        noisy = DecisionProblem(PayoffTable(randn(rng, 3, 80)),
                                SampleBelief((; x=randn(rng, 80)); weights=rand(rng, 80)))
        for method in (StrongOakley(), Sadatsafavi(), Grouping(; bins=8), SampleAverage())
            c = method isa SampleAverage ? evpi(noisy, method) : evppi(noisy, :x, method)
            @test c.value <= c.details.sample_evpi + 1e-12
            @test c.details.policy_gain ≈ c.value
            @test !c.details.above_sample_evpi
        end
        @test evpi(noisy, SampleAverage()).details.sample_evpi ≈ evpi(noisy)
    end

    @testset "Grouping refuses a continuous parameter without bins" begin
        rng = Xoshiro(5)
        n = 40
        belief = SampleBelief((; x=randn(rng, n), k=repeat(1:4, 10),
                               half=repeat([0.5, 1.5], 20)))
        problem = DecisionProblem(PayoffTable(randn(rng, 2, n)), belief)
        err = try
            evppi(problem, :x, Grouping())
        catch e
            e
        end
        @test err isa ArgumentError && occursin("x is continuous", err.msg)
        @test_throws ArgumentError evppi(problem, (:k, :x), Grouping())
        # With bins, or a parameter that does group the draws, it works as before.
        @test evppi(problem, :x, Grouping(; bins=4)).details.cells == 4
        @test evppi(problem, :k, Grouping()).details.cells == 4
        @test evppi(problem, :half, Grouping()).details.cells == 2
    end
end

@testset "SplineRegression chooses its basis from the draws" begin
    @test SplineRegression().automatic && SplineRegression(; knots=:auto).automatic
    @test !SplineRegression(; knots=4).automatic
    @test_throws ArgumentError SplineRegression(; knots=:many)
    @test_throws ArgumentError SplineRegression(; knots=-1)

    value_of(G) = sum(maximum(G; dims=1)) / size(G, 2) -
                  maximum(sum(G; dims=2)) / size(G, 2)
    function draws(rng, n, kind; noise)
        local x, z, Q
        x, z = randn(rng, n), randn(rng, n)
        signal = kind === :curved ? 0.2 .+ sin.(2 .* x) : fill(0.1, n)
        Q = vcat(zeros(1, n), permutedims(signal .+ noise .* randn(rng, n)),
                 permutedims(-0.1 .+ noise .* randn(rng, n)))
        return DecisionProblem(PayoffTable(Q), SampleBelief((; x, z))), x
    end

    # A parameter the payoffs do not depend on: four knots report value that is not
    # there, the chosen basis very little, and usually the constant, whose value is zero.
    fixed, chosen, constant = 0.0, 0.0, 0
    for seed in 1:100
        problem, _ = draws(Xoshiro(seed), 100, :none; noise=2.0)
        e = evppi(problem, :x)
        @test e.method === :SplineRegression
        @test e.details.degree == 0 ? iszero(e.value) : e.value >= 0
        constant += e.details.degree == 0
        chosen += e.value
        fixed += evppi(problem, :x, SplineRegression(; knots=4)).value
    end
    @test constant >= 60
    @test chosen < fixed / 4
    @test fixed / 100 > 0.15

    # A curved relationship with little noise: the choice keeps the curve.
    truth = let x = randn(Xoshiro(7), 1_000_000)
        value_of(vcat(zeros(1, length(x)), permutedims(0.2 .+ sin.(2 .* x)),
                      fill(-0.1, 1, length(x))))
    end
    errors = Float64[]
    for seed in 1:50
        problem, _ = draws(Xoshiro(1000 + seed), 200, :curved; noise=0.5)
        e = evppi(problem, :x)
        @test e.details.degree == 3 && e.details.knots >= 1
        push!(errors, e.value - truth)
        linear = evppi(problem, :x, SplineRegression(; knots=0, degree=1))
        @test linear.value < e.value
    end
    @test abs(sum(errors) / 50) < 0.02
    @test sqrt(sum(abs2, errors) / 50) < 0.06

    # The details carry every candidate, and the choice is the first smallest score.
    problem, x = draws(Xoshiro(3), 60, :curved; noise=1.0)
    e = evppi(problem, (:x, :z))
    scores = e.details.gcv
    @test [(s.degree, s.knots) for s in scores] ==
          [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]
    @test [s.coefficients for s in scores] == [1, 3, 5, 7, 9, 11, 13, 15]
    best = scores[argmin([s.gcv for s in scores])]
    @test (e.details.degree, e.details.knots) == (best.degree, best.knots)
    fixed_basis = SplineRegression(; knots=best.knots, degree=max(best.degree, 1))
    @test best.degree == 0 ||
          isapprox(e.value, evppi(problem, (:x, :z), fixed_basis).value; rtol=1e-9)

    # Candidates with more than one coefficient per two draws are left out.
    small, _ = draws(Xoshiro(4), 20, :curved; noise=0.5)
    few = evppi(small, (:x, :z)).details.gcv
    @test all(s -> 2 * s.coefficients <= 20, few) && length(few) == 5

    # A fixed basis has no choice to report; discrete columns are fitted as they are.
    @test !haskey(evppi(problem, :x, SplineRegression(; knots=4)).details, :gcv)
    g = repeat(1:4, 15)
    discrete = DecisionProblem(problem.payoff, SampleBelief((; g)))
    @test !haskey(evppi(discrete, :g, SplineRegression()).details, :gcv)
    @test evppi(discrete, :g, SplineRegression()).value ≈
          evppi(discrete, :g, Grouping()).value

    # Draws of zero weight do not enter the choice.
    n = 60
    keep = [isodd(i) for i in 1:n]
    Q = Float64.(problem.payoff.Q)
    weighted = DecisionProblem(PayoffTable(Q), SampleBelief((; x); weights=Float64.(keep)))
    subset = DecisionProblem(PayoffTable(Q[:, keep]), SampleBelief((; x=x[keep])))
    a, b = evppi(weighted, :x), evppi(subset, :x)
    @test (a.details.degree, a.details.knots) == (b.details.degree, b.details.knots)
    @test a.value ≈ b.value atol = 1e-8

    # One row (a quantity, not a decision): the score is of the row itself.
    quantity = (x .^ 2) .+ 0.1 .* randn(Xoshiro(5), n)
    reduction = expected_variance_reduction(SampleBelief((; x)), :x; quantity)
    @test 0 < reduction.reduction
end
