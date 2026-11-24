# The methods that follow the R package `voi` and Jackson et al. (2019, 2021, 2022):
# multivariate adaptive regression splines, two-level Monte Carlo, moment matching, the
# built-in study designs, value of information for estimation, the population value,
# the net benefit of sampling over sample sizes, and checks of a regression. Oracles:
# the closed forms of the normal model of the `voi` vignette, explicit loops, and
# `voi` itself through test/fixtures/voi/voi_extra_reference.json (written by
# validation/voi_extra_reference.R on the draws of normal_sample.csv).

@testset "methods following the R package voi" begin
    reference = JSON.parsefile(joinpath(@__DIR__, "fixtures", "voi",
                                        "voi_extra_reference.json"))
    shared = read_numeric_csv(joinpath(@__DIR__, "fixtures", "voi", "normal_sample.csv"))
    shared_problem = DecisionProblem(PayoffTable(permutedims(hcat(zero(shared.p1),
                                                                  shared.p1 .- shared.p2));
                                                 actions=[:reference, :new]),
                                     SampleBelief((; p1=shared.p1, p2=shared.p2)))
    payoffs(θ) = [0.0, θ.p1 - θ.p2]
    within(e, truth; slack=0.0) = abs(e.value - truth) < 4 * e.standard_error + slack
    evppi1, evppi2 = voi_normal_value(1.0), voi_normal_value(2.0)
    evsi_truth(m) = voi_normal_value(sqrt(m / (m + 1)))

    @testset "multivariate adaptive regression splines" begin
        @test reference["provenance"]["earth"] isa String
        problem = voi_normal_sample(Xoshiro(20261124), 20_000)
        e1 = evppi(problem, :p1, MARSRegression())
        e2 = evppi(problem, :p2, MARSRegression())
        @test within(e1, evppi1) && within(e2, evppi2)
        @test within(evppi(problem, (:p1, :p2), MARSRegression()),
                     voi_normal_value(sqrt(5)))
        @test e1.method === :MARSRegression
        @test first(e1.details.terms) == "1" && all(occursin("p1", t)
                                                    for t in e1.details.terms[2:end])
        @test e1.details.gcv > 0 && e1.details.residual_variance > 0
        # The fitted means keep the weighted mean of every action (an intercept).
        G = fitted_conditional_means(MARSRegression(), (; p1=problem.belief.draws.p1),
                                     problem.payoff.Q, problem.belief.weights)
        @test G * Float64.(problem.belief.weights) ≈
              problem.payoff.Q * Float64.(problem.belief.weights)
        # On the draws shared with R, `earth` (which may place a knot at any draw)
        # and this grid of knots are both within Monte Carlo error of the truth; the
        # two are not the same fit, and `earth`'s numbers are recorded, not matched.
        for (focal, key, truth) in ((:p1, "p1_earth", evppi1), (:p2, "p2_earth", evppi2))
            ours = evppi(shared_problem, focal, MARSRegression())
            @test within(ours, truth)
            @test abs(ours.value - reference["earth"][key]) < 5 * ours.standard_error
        end

        # A hinge is found exactly: the conditional mean is piecewise linear.
        rng = Xoshiro(8)
        n = 3000
        x = rand(rng, n)
        kink = 2 .* max.(x .- 0.3, 0.0) .- 0.4
        hinge = DecisionProblem(PayoffTable(permutedims(hcat(zeros(n),
                                                             kink .+ 0.05 .* randn(rng, n)))),
                                SampleBelief((; x)))
        exact = sum(max.(kink, 0.0)) / n - max(sum(kink) / n, 0.0)
        mars = evppi(hinge, :x, MARSRegression())
        @test mars.value ≈ exact atol = 0.004
        @test abs(mars.value - exact) <
              abs(evppi(hinge, :x, SplineRegression(; knots=0, degree=1)).value - exact)

        # An interaction needs degree 2: additive hinges cannot see a product.
        a, b = randn(rng, n), randn(rng, n)
        product = DecisionProblem(PayoffTable(permutedims(hcat(zeros(n),
                                                               a .* b .+
                                                               0.1 .* randn(rng, n)))),
                                  SampleBelief((; a, b)))
        target = sum(max.(a .* b, 0.0)) / n - max(sum(a .* b) / n, 0.0)
        @test evppi(product, (:a, :b), MARSRegression()).value < 0.3 * target
        two = evppi(product, (:a, :b), MARSRegression(; degree=2))
        @test two.value ≈ target rtol = 0.05
        @test any(occursin("·", t) for t in two.details.terms)

        # A non-numeric column enters as indicators; a constant payoff fits itself.
        site = [isodd(i) ? "wet" : "dry" for i in 1:n]
        by_site = DecisionProblem(PayoffTable(permutedims(hcat(zeros(n),
                                                               ifelse.(site .== "wet", 1.0,
                                                                       -1.0)))),
                                  SampleBelief((; site, x)))
        @test evppi(by_site, (:site, :x), MARSRegression()).value ≈
              evppi(by_site, :site, Grouping()).value
        flat = DecisionProblem(PayoffTable([zeros(n)'; ones(n)']), SampleBelief((; x)))
        @test evppi(flat, :x, MARSRegression()).value ≈ 0 atol = 1e-12
        # Weights: doubling a draw is the same as giving it twice the weight.
        small = 1:400
        doubled = vcat(small, 1:50)
        wide = DecisionProblem(PayoffTable(hinge.payoff.Q[:, doubled]),
                               SampleBelief((; x=x[doubled])))
        weighted = DecisionProblem(PayoffTable(hinge.payoff.Q[:, small]),
                                   SampleBelief((; x=x[small]);
                                                weights=[i <= 50 ? 2 : 1 for i in small]))
        @test evppi(wide, :x, MARSRegression(; knots=400)).value ≈
              evppi(weighted, :x, MARSRegression(; knots=400)).value
        @test_throws ArgumentError MARSRegression(; degree=0)
        @test_throws ArgumentError MARSRegression(; max_terms=0)
        @test_throws ArgumentError MARSRegression(; knots=0)
        @test_throws ArgumentError MARSRegression(; penalty=-1)
    end

    @testset "two-level Monte Carlo" begin
        focal1(rng) = (; p1=1 + randn(rng))
        rest1(rng, φ) = (; p2=2 * randn(rng))
        e = evppi_nested(payoffs, focal1, rest1; outer=3000, inner=300, seed=4)
        @test e.quantity === :evppi && e.method === :NestedMonteCarlo
        @test within(e, evppi1; slack=e.bias.cells)
        @test e.details.model_evaluations == 900_000
        # The inner noise inflates the estimate, and the reported bias says by how much:
        # few inner samples overstate, and more of them bring it down.
        crude = evppi_nested(payoffs, focal1, rest1; outer=3000, inner=4, seed=4)
        @test crude.value > e.value && crude.bias.cells > 5 * e.bias.cells
        @test abs(crude.value - crude.bias.cells - evppi1) < 6 * crude.standard_error
        # Reproducible; the other parameter; a named tuple of payoffs.
        @test evppi_nested(payoffs, focal1, rest1; outer=50, inner=20, seed=4).value ==
              evppi_nested(payoffs, focal1, rest1; outer=50, inner=20, seed=4).value
        e2 = evppi_nested(θ -> (; reference=0.0, new=θ.p1 - θ.p2),
                          rng -> (; p2=2 * randn(rng)), (rng, φ) -> (; p1=1 + randn(rng));
                          outer=3000, inner=200, seed=5)
        @test within(e2, evppi2; slack=e2.bias.cells)
        # Correlated parameters: the rest is drawn given the focal one. With
        # p2 = p1 / 2 + noise the payoff given p1 has mean p1 / 2, which is N(1/2, 1/4).
        tied = evppi_nested(payoffs, focal1, (rng, φ) -> (; p2=φ.p1 / 2 + randn(rng));
                            outer=3000, inner=300, seed=6)
        @test within(tied, expected_positive_part(0.5, 0.5) - 0.5; slack=tied.bias.cells)
        @test tied.value > 0.02
        @test_throws ArgumentError evppi_nested(payoffs, focal1, rest1; outer=10, inner=10)
        @test_throws ArgumentError evppi_nested(payoffs, focal1, rest1; outer=1, inner=10,
                                                seed=1)
        @test_throws ArgumentError evppi_nested(payoffs, rng -> 1.0, rest1; outer=5,
                                                inner=5, seed=1)
        @test_throws ArgumentError evppi_nested(θ -> "x", focal1, rest1; outer=5, inner=5,
                                                seed=1)
    end

    @testset "study designs" begin
        rng = Xoshiro(2)
        b = BinomialStudy(:p; n=40)
        xs = [simulate_study(rng, b, (; p=0.3)).x1 for _ in 1:4000]
        @test sum(xs) / 4000 ≈ 12 atol = 0.3
        @test sum(abs2, xs .- 12) / 4000 ≈ 40 * 0.3 * 0.7 atol = 0.6
        @test all(x -> 0 <= x <= 40, xs)
        # Log-likelihood up to a constant: differences are those of the binomial.
        @test study_loglikelihood(b, (; x1=10), (; p=0.3)) -
              study_loglikelihood(b, (; x1=10), (; p=0.2)) ≈
              10 * log(0.3 / 0.2) + 30 * log(0.7 / 0.8)
        @test study_loglikelihood(b, (; x1=0), (; p=0.0)) == 0
        @test study_loglikelihood(b, (; x1=3), (; p=0.0)) == -Inf
        # The conjugate posterior: Beta(a + x, b + n − x).
        post = study_posterior(rng, b, (; x1=10), (; a=2, b=3); draws=20_000).p
        α, β = 12, 33
        @test sum(post) / 20_000 ≈ α / (α + β) atol = 0.003
        @test sum(abs2, post .- α / (α + β)) / 20_000 ≈
              α * β / ((α + β)^2 * (α + β + 1)) rtol = 0.05
        # A Beta with a shape below one (the boost of the gamma sampler).
        sparse = study_posterior(rng, BinomialStudy(:p; n=1), (; x1=0), (; a=0.3, b=0.2);
                                 draws=20_000).p
        @test sum(sparse) / 20_000 ≈ 0.3 / 1.5 atol = 0.006

        t = TwoArmBinomialStudy(:control, :treated; n=25)
        data = simulate_study(rng, t, (; control=0.2, treated=0.6))
        @test keys(data) == (:x1, :x2)
        @test study_loglikelihood(t, (; x1=5, x2=15), (; control=0.2, treated=0.6)) ≈
              5 * log(0.2) + 20 * log(0.8) + 15 * log(0.6) + 10 * log(0.4)
        two = study_posterior(rng, t, (; x1=5, x2=15), (; a1=1, b1=1, a2=1, b2=1);
                              draws=10_000)
        @test keys(two) == (:control, :treated)
        @test sum(two.treated) / 10_000 ≈ 16 / 27 atol = 0.005

        m = NormalStudy(:mu; n=16, sd=2)
        bars = [simulate_study(rng, m, (; mu=1.5)).xbar for _ in 1:4000]
        @test sum(bars) / 4000 ≈ 1.5 atol = 0.03
        @test sum(abs2, bars .- 1.5) / 4000 ≈ 4 / 16 rtol = 0.08
        @test study_loglikelihood(m, (; xbar=1.0), (; mu=1.5)) ≈ -16 * 0.25 / 8
        normal = study_posterior(rng, m, (; xbar=1.0), (; mean=0.0, sd=1.0);
                                 draws=20_000).mu
        @test sum(normal) / 20_000 ≈ (16 / 4) / (1 + 16 / 4) atol = 0.01
        @test sum(abs2, normal .- 0.8) / 20_000 ≈ 1 / 5 rtol = 0.05

        @test_throws ArgumentError BinomialStudy(:p; n=0)
        @test_throws ArgumentError TwoArmBinomialStudy(:p, :p; n=5)
        @test_throws ArgumentError NormalStudy(:mu; n=5, sd=0)
        @test_throws ArgumentError simulate_study(rng, b, (; p=1.2))
        @test_throws ArgumentError simulate_study(rng, b, (; q=0.2))
        @test_throws ArgumentError study_posterior(rng, b, (; x1=1), (; mean=0, sd=1))
        @test_throws ArgumentError study_posterior(rng, b, (; x1=1), (; a=-2, b=1))

        # The same design through the regression and importance-sampling routes.
        problem = voi_normal_sample(Xoshiro(31), 8000)
        for size in (10, 100)
            design = NormalStudy(:p1; n=size)
            regression = evsi(problem, SummaryStatistics(problem, design; seed=3))
            @test within(regression, evsi_truth(size))
            @test regression.details.summaries == [:xbar]
            importance = evsi(problem,
                              ImportanceSampling(problem.belief, design; seed=3,
                                                 datasets=600))
            @test within(importance, evsi_truth(size); slack=0.01)
        end
        @test_throws ArgumentError SummaryStatistics(problem, NormalStudy(:p1; n=5))
        @test SummaryStatistics(problem, NormalStudy(:p1; n=5); seed=1).summaries ==
              SummaryStatistics(problem, NormalStudy(:p1; n=5); seed=1).summaries

        # A binomial study on a problem whose answer is known by enumeration: the
        # prior is a Beta, the payoff of acting is linear in the probability, so the
        # preposterior mean is linear in the count.
        n = 30_000
        rng = Xoshiro(12)
        p = [DecisionAnalysis._beta(rng, 3, 5) for _ in 1:n]
        linear = DecisionProblem(PayoffTable(permutedims(hcat(zeros(n), p .- 0.4));
                                             actions=[:none, :act]), SampleBelief((; p)))
        size = 12
        # P(x) under the beta-binomial, and the posterior mean (3 + x) / (8 + n).
        lbeta(a, b) = sum(log, 1:(a - 1); init=0.0) + sum(log, 1:(b - 1); init=0.0) -
                      sum(log, 1:(a + b - 1); init=0.0)
        px(x) = binomial(size, x) * exp(lbeta(3 + x, 5 + size - x) - lbeta(3, 5))
        exact = sum(px(x) * max((3 + x) / (8 + size) - 0.4, 0.0) for x in 0:size) -
                max(3 / 8 - 0.4, 0.0)
        design = BinomialStudy(:p; n=size)
        @test within(evsi(linear, SummaryStatistics(linear, design; seed=1)), exact)
        @test within(evsi(linear, ImportanceSampling(linear, design; seed=1, datasets=800)),
                     exact; slack=0.002)
        mm = evsi(linear,
                  MomentMatching(design; payoffs=θ -> [0.0, θ.p - 0.4], prior=(; a=3, b=5)))
        @test within(mm, exact; slack=0.002)
    end

    @testset "moment matching" begin
        problem = voi_normal_sample(Xoshiro(20261125), 20_000)
        values = Float64[]
        for size in (10, 100)
            study = MomentMatching(NormalStudy(:p1; n=size); payoffs,
                                   prior=(; mean=1.0, sd=1.0))
            e = evsi(problem, study)
            push!(values, e.value)
            @test e.quantity === :evsi && e.method === :MomentMatching
            @test within(e, evsi_truth(size))
            # The share of the EVPPI variance that the study removes: n / (n + 1).
            @test e.details.variance_ratio ≈ size / (size + 1) atol = 0.08
            # The model is run twice per posterior draw: prior and posterior focal values.
            @test e.details.model_evaluations == 2 * 30 * 1000
            @test e.standard_error ≈
                  hypot(e.details.draws_standard_error, e.details.nested_standard_error)
            @test e.details.capped == (e.details.variance_ratio > 1) &&
                  e.details.focal == [:p1]
            @test e.value <= evppi(problem, :p1).value + 1e-12
            # `voi` on the shared draws, with its own nested sample: recorded with
            # the truth beside it, since the two differ by Monte Carlo error.
            @test abs(reference["moment_matching"]["evsi_p1_n$(size)_mm"] -
                      evsi_truth(size)) < 0.03
        end
        @test issorted(values)
        # The same through the functions of a study of one's own.
        own = MomentMatching(:p1; payoffs, nested=20,
                             simulate_data=(rng, θ) -> θ.p1 + randn(rng) / sqrt(10),
                             posterior=(rng, x) -> (;
                                                    p1=(1 + 10x) / 11 .+
                                                       randn(rng, 500) ./ sqrt(11)))
        e = evsi(problem, own; seed=3)
        @test within(e, evsi_truth(10)) && e.details.nested == 20
        @test evsi(problem, own; seed=3).value == e.value
        # A study of a parameter that the decision does not depend on is worth nothing.
        idle = DecisionProblem(problem.payoff,
                               SampleBelief(merge(problem.belief.draws,
                                                  (; z=randn(Xoshiro(1), 20_000)))))
        nothing_learned = evsi(idle,
                               MomentMatching(NormalStudy(:z; n=10); payoffs,
                                              prior=(; mean=0.0, sd=1.0),
                                              regression=Grouping(; bins=1)))
        @test iszero(nothing_learned.value) && iszero(nothing_learned.standard_error)
        # Three actions: the shrinkage is a matrix, checked against the regression on
        # the sufficient statistic.
        rng = Xoshiro(9)
        n = 20_000
        p1, p2 = 1 .+ randn(rng, n), 2 .* randn(rng, n)
        three = DecisionProblem(PayoffTable(permutedims(hcat(zeros(n), p1 .- p2,
                                                             0.5 .- 0.5 .* p1 .+ 0.2 .* p2));
                                            actions=[:a, :b, :c]), SampleBelief((; p1, p2)))
        design = NormalStudy(:p1; n=10)
        direct = evsi(three, SummaryStatistics(three, design; seed=2))
        matched = evsi(three,
                       MomentMatching(design; prior=(; mean=1.0, sd=1.0),
                                      payoffs=θ -> [0.0, θ.p1 - θ.p2,
                                                    0.5 - 0.5 * θ.p1 + 0.2 * θ.p2]))
        @test abs(matched.value - direct.value) <
              4 * hypot(matched.standard_error, direct.standard_error)
        @test matched.value > 0
        @test_throws ArgumentError MomentMatching(:p1; payoffs, simulate_data=identity,
                                                  posterior=identity, nested=2)
        @test_throws ArgumentError evsi(problem,
                                        MomentMatching(NormalStudy(:p1; n=5);
                                                       payoffs=θ -> [0.0],
                                                       prior=(; mean=1.0, sd=1.0)))
        @test_throws ArgumentError evsi(problem,
                                        MomentMatching(:p1; payoffs,
                                                       simulate_data=(rng, θ) -> 0.0,
                                                       posterior=(rng, x) -> (;
                                                                              q=[1.0, 2.0])))
        @test_throws BeliefFormError evsi(DecisionProblem(PayoffTable([0 1; 1 0]),
                                                          [1 // 2, 1 // 2]),
                                          MomentMatching(NormalStudy(:p1; n=5); payoffs,
                                                         prior=(; mean=1.0, sd=1.0)))
    end

    @testset "value of information for estimation" begin
        belief = shared_problem.belief
        y = shared.p1 .- shared.p2
        v = reference["variance"]
        n = length(y)
        r1 = expected_variance_reduction(belief, :p1; quantity=y)
        @test r1.prior_variance ≈ v["var_y"] * (n - 1) / n
        # Against `voi` on the same draws (its spline is penalised, ours is not), and
        # against the closed forms 1 and 4.
        @test r1.reduction ≈ v["evppivar_p1_gam"] rtol = 0.02
        @test expected_variance_reduction(belief, :p2; quantity=y).reduction ≈
              v["evppivar_p2_gam"] rtol = 0.02
        @test expected_variance_reduction(belief, :p1; quantity=y,
                                          method=MARSRegression()).reduction ≈
              v["evppivar_p1_earth"] rtol = 0.03
        @test r1.reduction ≈ 1 atol = 0.1
        @test r1.share ≈ r1.reduction / r1.prior_variance
        @test r1.expected_posterior_variance ≈ r1.prior_variance - r1.reduction
        # The problem and its belief give the same answer; a column by name.
        @test expected_variance_reduction(shared_problem, :p1; quantity=y) == r1
        @test expected_variance_reduction(belief, :p2; quantity=:p1).share < 0.02
        @test expected_variance_reduction(belief, :p1; quantity=:p1).share ≈ 1

        # Studies: n observations of p1 remove n / (n + 1) of what p1 explains.
        for size in (10, 100)
            design = NormalStudy(:p1; n=size)
            truth = size / (size + 1)
            regression = expected_variance_reduction(belief,
                                                     SummaryStatistics(belief, design;
                                                                       seed=4); quantity=y)
            @test regression.reduction ≈ truth atol = 0.12
            @test regression.reduction ≈ v["evsivar_p1_n$(size)_gam"] atol = 0.12
            approximate = expected_variance_reduction(belief,
                                                      GaussianApproximation(:p1; n0=1,
                                                                            n=size);
                                                      quantity=y)
            @test approximate.reduction ≈ r1.reduction * truth
            weighted = expected_variance_reduction(shared_problem,
                                                   ImportanceSampling(belief, design;
                                                                      seed=4, datasets=500);
                                                   quantity=y)
            @test weighted.reduction ≈ truth atol = 0.15
            @test weighted.prior_variance ≈ r1.prior_variance
        end
        curve = expected_variance_reduction(belief,
                                            GaussianApproximation(:p1; n0=1,
                                                                  n=[1, 10, 100]);
                                            quantity=y)
        @test issorted([c.reduction for c in curve]) &&
              curve[1].reduction ≈ r1.reduction / 2

        # Importance sampling against explicit enumeration on a tiny sample.
        tiny = SampleBelief((; x=[0.0, 1.0, 3.0]); weights=[1, 2, 1])
        L = log.([0.5 0.3 0.2; 0.1 0.3 0.6])
        g = [1.0, 2.0, 5.0]
        w = [0.25, 0.5, 0.25]
        post(k) = (w .* exp.(L[k, :])) ./ sum(w .* exp.(L[k, :]))
        remaining = sum(0.5 * sum(post(k) .* (g .- sum(post(k) .* g)) .^ 2) for k in 1:2)
        total = sum(w .* (g .- sum(w .* g)) .^ 2)
        r = expected_variance_reduction(tiny, ImportanceSampling(L; weights=:equal);
                                        quantity=g)
        @test r.prior_variance ≈ total && r.expected_posterior_variance ≈ remaining

        # Inputs by outputs.
        table = variance_reduction_table(belief, [:p1, :p2, (:p1, :p2)],
                                         (; difference=y, total=shared.p1 .+ shared.p2,
                                          p1=:p1))
        @test table.inputs == ["p1", "p2", "p1, p2"]
        @test table.outputs == [:difference, :total, :p1]
        @test size(table.share) == (3, 3)
        @test table.share[1, 1] == r1.share
        @test table.share[3, :] ≈ ones(3) atol = 0.01      # both inputs: everything
        @test table.share[1, 3] ≈ 1 && table.share[2, 3] < 0.02
        @test table.remaining_sd ≈
              sqrt.(max.(permutedims(table.prior_variance) .- table.reduction, 0.0))
        @test variance_reduction_table(belief, :p1, [:p2]).outputs == [:p2]
        @test_throws ArgumentError variance_reduction_table(belief, Symbol[], (; y))
        @test_throws ArgumentError expected_variance_reduction(belief, :p1; quantity=[1.0])
        @test_throws ArgumentError expected_variance_reduction(belief, :p1; quantity=y,
                                                               method=SampleAverage())
        @test_throws BeliefFormError expected_variance_reduction(DecisionProblem(PayoffTable([0 1;
                                                                                              1 0]),
                                                                                 [1 // 2,
                                                                                  1 // 2]),
                                                                 SummaryStatistics((;
                                                                                    s=[1.0,
                                                                                       2.0]));
                                                                 quantity=[1.0, 2.0])

        @test equivalent_binomial_sample_size(0.2, 0.1) == 15
        @test equivalent_binomial_sample_size(0.2, 0.01) == 1599
        # A Beta(a, b) with a + b = n has that mean and standard deviation.
        @test equivalent_binomial_sample_size(3 / 8, sqrt(15 / (64 * 9))) ≈ 8
        @test_throws ArgumentError equivalent_binomial_sample_size(1.2, 0.1)
        @test_throws ArgumentError equivalent_binomial_sample_size(0.5, 0.0)
        @test_throws ArgumentError equivalent_binomial_sample_size(0.5, 0.6)
    end

    @testset "population value and the net benefit of sampling" begin
        d = reference["design"]
        @test population_value(2; population=10) == 20
        @test population_value(2; population=10, horizon=5) == 100
        @test population_value(2.0; population=10, horizon=5, discount=0.035) ≈
              d["pop_voi_dis"]
        # `voi` returns value × population at a zero rate; the limit of its own formula
        # is value × population × horizon, which is what is returned here.
        @test d["pop_voi_zero"] == 20
        @test population_value(2.0; population=10, horizon=5, discount=1e-9) ≈ 100 rtol = 1e-6
        estimate = evsi(shared_problem, GaussianApproximation(:p1; n0=1, n=10))
        scaled = population_value(estimate; population=50)
        @test scaled.value == 50 * estimate.value &&
              scaled.standard_error == 50 * estimate.standard_error
        @test_throws ArgumentError population_value(1; population=-1)
        @test_throws ArgumentError population_value(1; population=1, horizon=-1)
        @test_throws ArgumentError population_value(1; population=1, discount=-0.1)

        # The curve of `voi::enbs` on a fixed EVSI curve.
        sizes = Int.(d["n"])
        curve = enbs_curve(Float64.(d["evsi"]); n=sizes, setup_cost=(5, 15),
                           per_sample_cost=(0.04, 0.06), population=1000, horizon=10,
                           discount=0.035)
        @test [row.enbs for row in curve] ≈ d["enbs"] rtol = 1e-9
        @test [row.standard_deviation for row in curve] ≈ d["sd"] rtol = 1e-9
        @test [row.probability_positive for row in curve] ≈ d["pce"] atol = 1e-6
        @test all(row.standard_error === nothing for row in curve)
        @test [row.cost for row in curve] == 10 .+ 0.05 .* sizes
        best = optimal_sample_size(curve)
        @test best.n == d["nmax"] && best.enbs ≈ d["enbsmax"]
        @test best.lower == d["nlower"] && best.upper == d["nupper"] && !best.smoothed
        smooth = optimal_sample_size(curve; smooth=true)
        @test smooth.n >= 250 && smooth.smoothed
        @test smooth.enbs ≈ d["smooth_enbsmax"] rtol = 0.01
        @test abs(smooth.lower - d["smooth_nlower"]) <= 25 &&
              smooth.upper == d["smooth_nupper"]

        # An interior optimum, recovered from noisy estimates by smoothing.
        sizes = collect(5:5:200)
        exact = enbs_curve(evsi_truth.(sizes); n=sizes, setup_cost=10, per_sample_cost=0.05,
                           population=1000)
        every = collect(5:200)
        truth = optimal_sample_size(enbs_curve(evsi_truth.(every); n=every, setup_cost=10,
                                               per_sample_cost=0.05, population=1000))
        @test 40 <= truth.n <= 55
        @test abs(optimal_sample_size(exact; smooth=true).n - truth.n) <= 8
        rng = Xoshiro(3)
        noisy = [merge(row, (; enbs=row.enbs + randn(rng))) for row in exact]
        recovered = optimal_sample_size(noisy; smooth=true)
        @test abs(recovered.n - truth.n) <= 30
        @test recovered.lower <= truth.n <= recovered.upper
        @test recovered.lower < recovered.n < recovered.upper
        # Certain costs: no probability; a three-number cost; estimates carry errors.
        @test exact[1].probability_positive === nothing &&
              iszero(exact[1].standard_deviation)
        three = enbs_curve([0.1]; n=[10], setup_cost=(12, 5, 15))
        @test three[1].cost == 12 && three[1].standard_deviation == 2.5
        @test three[1].probability_positive < 1e-5
        estimates = evsi(shared_problem, GaussianApproximation(:p1; n0=1, n=[10, 100]))
        with_errors = enbs_curve(estimates; n=[10, 100], population=20)
        @test with_errors[2].standard_error == 20 * estimates[2].standard_error
        @test with_errors[2].enbs == 20 * estimates[2].value
        @test_throws ArgumentError enbs_curve([0.1, 0.2]; n=[10])
        @test_throws ArgumentError enbs_curve([0.1]; n=[10], setup_cost=(3, 1))
        @test_throws ArgumentError enbs_curve([0.1]; n=[10], per_sample_cost="x")
        @test_throws ArgumentError enbs_curve([evppi(shared_problem, :p1)]; n=[10])
        @test_throws ArgumentError optimal_sample_size(exact; near=1.5)
        @test_throws ArgumentError optimal_sample_size(exact[1:3]; smooth=true)
        @test_throws ArgumentError optimal_sample_size(NamedTuple[])
    end

    @testset "checks of a regression" begin
        problem = voi_normal_sample(Xoshiro(44), 6000)
        good = regression_check(problem, :p1, SplineRegression())
        @test good.method === :SplineRegression && length(good.bins) == 10
        @test good.worst < 3.5                                   # residuals centred
        @test good.r_squared ≈ 1 / 5 atol = 0.03                 # p1 explains 1 of 5
        # The act-now action is `new` (its mean is 1), so the check is of `reference`.
        @test all(row.action === :reference for row in good.bins)
        @test issorted([row.fitted for row in good.bins])
        # A fit that cannot follow the shape fails the check where it misses.
        rng = Xoshiro(6)
        n = 4000
        x = rand(rng, n)
        bent = DecisionProblem(PayoffTable(permutedims(hcat(zeros(n),
                                                            4 .* (x .- 0.5) .^ 2 .- 0.3 .+
                                                            0.1 .* randn(rng, n)));
                                           actions=[:none, :act]), SampleBelief((; x)))
        straight = SplineRegression(; knots=0, degree=1)
        @test regression_check(bent, :x, straight).worst > 8
        @test regression_check(bent, :x, SplineRegression()).worst < 3.5
        @test regression_check(bent, :x, MARSRegression(); bins=5).worst < 5
        # The study form, and the bins as explicit weighted means.
        study = SummaryStatistics(problem, NormalStudy(:p1; n=10); seed=1)
        @test regression_check(problem, study).worst < 3.5
        check = regression_check(bent, :x, straight; bins=4)
        G = fitted_conditional_means(straight, (; x), bent.payoff.Q, bent.belief.weights)
        # The act-now action is `act` (its mean payoff is positive), so the checked
        # difference is that of `none` from it.
        @test all(row.action === :none for row in check.bins)
        fitted = G[1, :] .- G[2, :]
        residual = (bent.payoff.Q[1, :] .- bent.payoff.Q[2, :]) .- fitted
        order = sortperm(fitted)
        quarter = order[1:(n ÷ 4)]
        @test check.bins[1].residual_mean ≈ sum(residual[quarter]) / length(quarter)
        @test check.bins[1].fitted ≈ sum(fitted[quarter]) / length(quarter)
        @test_throws ArgumentError regression_check(problem, :p1, SampleAverage())
        @test_throws ArgumentError regression_check(problem, :p1, SplineRegression();
                                                    bins=1)

        # Cross-validation prefers the method that predicts held-out draws best: the
        # straight line loses on the bent problem, and wins nothing on the normal one.
        candidates = (; straight, spline=SplineRegression(), mars=MARSRegression(),
                      blocks=StrongOakley())
        compared = compare_regressions(bent, :x, candidates; folds=5, seed=2)
        rows = Dict(row.name => row for row in compared.methods)
        @test compared.best in (:spline, :mars)
        @test rows[:straight].cv_r_squared < 0.1 < rows[:spline].cv_r_squared
        @test rows[:straight].cv_error > 5 * rows[:spline].cv_error
        @test rows[:spline].value == evppi(bent, :x, SplineRegression()).value
        @test !rows[:spline].above_sample_evpi
        # Held-out error is at least the in-sample error of the same fit.
        inside = regression_check(bent, :x, SplineRegression())
        @test rows[:spline].cv_error >= inside.residual_variance - 1e-12
        # An overfitted regression: flagged, and worse out of sample than blocks that
        # report nothing to learn. No warning leaks from the comparison.
        m = 200
        z = collect(1:m) ./ m
        step = DecisionProblem(PayoffTable([zeros(m)';
                                            [0.4 < v <= 0.5 ? -40.0 : -1.0 for v in z]'];
                                           actions=[:none, :act]), SampleBelief((; z)))
        overfit = @test_logs compare_regressions(step, :z,
                                                 (; spline=SplineRegression(),
                                                  blocks=StrongOakley()))
        byname = Dict(row.name => row for row in overfit.methods)
        @test byname[:spline].above_sample_evpi && !byname[:blocks].above_sample_evpi
        @test iszero(byname[:blocks].value)
        # Reproducible; other forms of the list; a study; bad input.
        @test compare_regressions(bent, :x, candidates; seed=2).methods ==
              compared.methods
        listed = compare_regressions(bent, :x, [straight, MARSRegression()])
        @test [row.name for row in listed.methods] == [:SplineRegression, :MARSRegression]
        @test compare_regressions(problem, study, SplineRegression()).best ===
              :SplineRegression
        @test_throws ArgumentError compare_regressions(bent, :x,
                                                       [SplineRegression(),
                                                        SplineRegression(; knots=1)])
        @test_throws ArgumentError compare_regressions(bent, :x, (; a=Sadatsafavi()))
        @test_throws ArgumentError compare_regressions(bent, :x, candidates; folds=1)
        @test_throws ArgumentError compare_regressions(bent, :x, ())
    end
end
