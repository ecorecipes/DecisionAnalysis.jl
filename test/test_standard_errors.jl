# Honest standard errors, and the exhaustive Sadatsafavi search. The oracle for a
# standard error is the spread of the estimate over independent repetitions of the
# whole computation (fresh draws, fresh simulated data), which shares no code with
# the formula inside the estimator.

function spread(values)
    m = sum(values) / length(values)
    return sqrt(sum(v -> (v - m)^2, values) / (length(values) - 1))
end

@testset "standard errors and the Sadatsafavi search" begin
    @testset "importance sampling: both sources of error" begin
        m = 10
        truth = voi_normal_value(sqrt(m / (m + 1)))
        simulate_data(rng, θ) = θ.p1 + randn(rng) / sqrt(m)
        loglikelihood(x, θ) = -m * (x - θ.p1)^2 / 2
        # Two designs: few data sets (their error dominates) and many (the error from
        # the draws dominates, which the old standard error left out).
        for (n, K) in ((1500, 80), (400, 600))
            rng = Xoshiro(81)
            estimates = map(1:80) do _
                problem = voi_normal_sample(rng, n)
                study = importance_loglikelihoods(simulate_data, loglikelihood,
                                                  problem.belief; datasets=K,
                                                  seed=rand(rng, 1:(10 ^ 9)))
                return evsi(problem, study)
            end
            values = [e.value for e in estimates]
            reported = sum(e.standard_error for e in estimates) / length(estimates)
            from_datasets = sum(e.details.datasets_standard_error for e in estimates) /
                            length(estimates)
            from_draws = sum(e.details.draws_standard_error for e in estimates) /
                         length(estimates)
            empirical = spread(values)
            # The reported error matches the spread over repetitions ...
            @test 0.75 < reported / empirical < 1.35
            # ... and neither source alone accounts for it. With many data sets the
            # error from the data sets, which used to be the whole reported standard
            # error, is well under half of the spread.
            @test from_datasets < reported && from_draws < reported
            @test K > n ? from_draws > from_datasets : from_datasets > from_draws
            K > n && @test from_datasets < 0.5 * empirical
            # The estimate is centred on the closed form, within the error of the mean.
            @test abs(sum(values) / length(values) - truth) <
                  4 * empirical / sqrt(length(values)) + 0.005
            for e in estimates[1:5]
                @test e.standard_error ≈ hypot(e.details.datasets_standard_error,
                                               e.details.draws_standard_error)
                # The effective sample sizes are in the estimate, and printed with it.
                d = e.details
                @test 1 <= d.minimum_ess <= d.median_ess <= n
                @test d.minimum_ess <= d.mean_ess <= n
                @test d.prior_ess ≈ n
                @test occursin("posterior effective sample size", sprint(show, e))
            end
        end

        # The two terms from their definitions, on a small exact problem: every signal
        # of a finite experiment as a data set with its probability as weight.
        Q = Float64[0 0 0 0; 3 1 -4 2; -2 2 1 -1]
        w = [0.4, 0.3, 0.2, 0.1]
        L = [0.7 0.2 0.1 0.5; 0.2 0.6 0.2 0.3; 0.1 0.2 0.7 0.2]
        v = [sum(w[j] * L[k, j] for j in 1:4) for k in 1:3]
        problem = DecisionProblem(PayoffTable(Q), SampleBelief(4; weights=w))
        e = evsi(problem, ImportanceSampling(log.(L); weights=v))
        a0 = first(argmax_set(oracle_expected_utility(Q, w)))
        gains = zeros(3)
        influence = zeros(4)
        for k in 1:3
            posterior = [w[j] * L[k, j] / v[k] for j in 1:4]
            scores = oracle_expected_utility(Q, posterior)
            b = first(argmax_set(scores))
            gains[k] = scores[b] - scores[a0]
            for j in 1:4
                influence[j] += v[k] * posterior[j] * (Q[b, j] - Q[a0, j] - gains[k])
            end
        end
        mean_gain = sum(v .* gains)
        @test e.value ≈ mean_gain ≈ oracle_evsi(Q, w, L)
        @test e.details.datasets_standard_error ≈
              sqrt(sum(v[k]^2 * (gains[k] - mean_gain)^2 for k in 1:3))
        @test e.details.draws_standard_error ≈ sqrt(sum(abs2, influence))
        # No information: every posterior is the prior, nothing is gained, no error.
        flat = evsi(problem, ImportanceSampling(zeros(3, 4); weights=:equal))
        @test flat.value ≈ 0 atol = 1e-12
        @test flat.standard_error ≈ 0 atol = 1e-12
    end

    @testset "Gaussian approximation: linearised error against repetitions and bootstrap" begin
        # The limits stated in the docstring of `GaussianApproximation`: with the
        # default regression the linearised standard error is close to the spread over
        # repetitions and to the bootstrap; with a linear fit it is conservative.
        m, n = 10, 1000
        function repetitions(regression, count; bootstrap=0)
            rng = Xoshiro(82)
            return map(1:count) do _
                problem = voi_normal_sample(rng, n)
                return evsi(problem, GaussianApproximation(:p1; n0=1, n=m, regression);
                            bootstrap, seed=rand(rng, 1:(10 ^ 6)))
            end
        end
        default = repetitions(nothing, 150)
        empirical = spread([e.value for e in default])
        reported = sum(e.standard_error for e in default) / length(default)
        @test 0.85 < reported / empirical < 1.25
        boot = repetitions(nothing, 12; bootstrap=60)
        @test [e.value for e in boot] == [e.value for e in default[1:12]]
        bootstrap = sum(e.standard_error for e in boot) / length(boot)
        @test 0.7 < bootstrap / empirical < 1.3
        @test 0.7 < bootstrap / reported < 1.3
        linear = repetitions(SplineRegression(; knots=0, degree=1), 150)
        empirical = spread([e.value for e in linear])
        reported = sum(e.standard_error for e in linear) / length(linear)
        @test 1.0 < reported / empirical < 1.5         # conservative, by about a quarter
    end

    @testset "Sadatsafavi: the search is exhaustive" begin
        # Brute force over every pair of cuts, with the best action chosen per
        # segment, against the running-maximum search; three actions, so that the
        # middle segment matters.
        rng = Xoshiro(83)
        for _ in 1:25
            n, m = rand(rng, 12:30), rand(rng, 2:4)
            Q = BigRat.(rand(rng, -9:9, m, n))
            x = Random.shuffle(rng, collect(1:n))
            problem = DecisionProblem(PayoffTable(Q), SampleBelief((; x)))
            order = sortperm(x)
            S = Q[:, order]
            best_in(from, to) = maximum(sum(S[a, j] for j in from:to) for a in 1:m)
            act_now = best_in(1, n)
            one = maximum(best_in(1, s) + best_in(s + 1, n) for s in 1:(n - 1))
            two = maximum(best_in(1, s) + best_in(s + 1, t) + best_in(t + 1, n)
                          for s in 1:(n - 2) for t in (s + 1):(n - 1))
            one, two = max(one, act_now), max(one, two, act_now)
            @test evppi(problem, :x, Sadatsafavi()).value ≈ Float64((one - act_now) // n) atol = 1e-9
            @test evppi(problem, :x, Sadatsafavi(; separators=2)).value ≈
                  Float64((two - act_now) // n) atol = 1e-9
            @test two >= one
        end
        # A large sample: two separators are never worse than one (the search used to
        # be restricted to a grid of positions, which could make them worse).
        large = voi_normal_sample(Xoshiro(84), 5000)
        for focal in (:p1, :p2)
            @test evppi(large, focal, Sadatsafavi(; separators=2)).value >=
                  evppi(large, focal, Sadatsafavi()).value - 1e-12
        end
    end
end
