# Gaussian-process EVPPI, checked as the other regressions are: against exact enumeration
# on finite problems turned into samples, and against closed forms of normal models.

# The normal model of the R `voi` vignette extended to six independent parameters:
# two actions with payoffs 0 and 1 + Σ_j c_j x_j, x_j ~ N(0, 1). Learning the
# parameters in S leaves X_S ~ N(1, Σ_{j ∈ S} c_j²), so the EVPPI of S is
# E max(0, X_S) − 1 in closed form.
const SIX_COEFFICIENTS = [1.0, 2.0, 0.5, 0.5, 0.5, 0.5]
function six_parameter_sample(rng, n)
    X = randn(rng, n, 6)
    new = 1 .+ X * SIX_COEFFICIENTS
    draws = NamedTuple{Tuple(Symbol.("x", 1:6))}(Tuple(X[:, j] for j in 1:6))
    return DecisionProblem(PayoffTable(permutedims(hcat(zeros(n), new));
                                       actions=[:reference, :new]), SampleBelief(draws))
end
function six_parameter_value(indices)
    return expected_positive_part(1.0, sqrt(sum(abs2, SIX_COEFFICIENTS[indices]))) - 1
end

@testset "Gaussian-process EVPPI" begin
    @testset "the method is chosen by the number of focal parameters" begin
        n = 60
        rng = Xoshiro(30)
        draws = merge(NamedTuple{Tuple(Symbol.("x", 1:6))}(Tuple(randn(rng, n) for _ in 1:6)),
                      (; site=rand(rng, ["wet", "dry"], n)))
        problem = DecisionProblem(PayoffTable(randn(rng, 2, n)), SampleBelief(draws))
        # As the R package `voi`: smooths up to four parameters, a Gaussian process from
        # five; a non-numeric parameter is grouped.
        @test evppi(problem, :x1).method === :SplineRegression
        @test evppi(problem, (:x1, :x2, :x3, :x4)).method === :SplineRegression
        @test evppi(problem, (:x1, :x2, :x3, :x4, :x5)).method ===
              :GaussianProcessRegression
        @test evppi(problem, (:x1, :x2, :x3, :x4, :x5, :x6)).method ===
              :GaussianProcessRegression
        @test evppi(problem, :site).method === :Grouping
        # A continuous parameter with a non-numeric one: grouping would make every draw
        # a cell, so the spline regression takes the non-numeric one as a factor.
        @test evppi(problem, (:x1, :site)).method === :SplineRegression
        @test_throws ArgumentError evppi(problem, (:x1, :site), Grouping())
        # An explicit method overrides the rule.
        @test evppi(problem, :x1, GaussianProcessRegression()).method ===
              :GaussianProcessRegression
        @test evsi(problem,
                   GaussianApproximation((:x1, :x2, :x3, :x4, :x5); n0=1, n=5)).details.fit ===
              :GaussianProcessRegression
        @test_throws ArgumentError evppi(problem, (:x1, :site), GaussianProcessRegression())
        @test_throws ArgumentError GaussianProcessRegression(; max_points=5)
        @test_throws ArgumentError GaussianProcessRegression(; hyper_points=5)
        @test_throws ArgumentError GaussianProcessRegression(; min_nugget=0)
        @test_throws ArgumentError GaussianProcessRegression(; min_length_scale=0)
        few = DecisionProblem(PayoffTable(randn(rng, 2, 6)),
                              SampleBelief((; a=randn(rng, 6), b=randn(rng, 6),
                                            c=randn(rng, 6))))
        @test_throws ArgumentError evppi(few, (:a, :b, :c), GaussianProcessRegression())
    end

    @testset "exact enumeration" begin
        rng = Xoshiro(31)
        for _ in 1:5
            # Every state is a distinct point of five parameters and the payoffs are
            # smooth functions of them: learning all five is perfect information, and a
            # process with a negligible nugget interpolates, so the estimate is the EVPI
            # by enumeration.
            n, m = 80, 3
            X = [rand(rng, -5:5) for _ in 1:n, _ in 1:5]
            while length(unique(eachrow(X))) < n
                X = [rand(rng, -5:5) for _ in 1:n, _ in 1:5]
            end
            Q = zeros(BigRat, m, n)
            for a in 1:m
                b, c, pair = rand(rng, -3:3, 5), rand(rng, -2:2), rand(rng, 1:5, 2)
                for i in 1:n
                    Q[a, i] = sum(b .* X[i, :]) + c * X[i, pair[1]] * X[i, pair[2]] // 2
                end
            end
            w = rand(rng, 1:4, n)
            p = BigRat.(w) .// sum(w)
            draws = NamedTuple{Tuple(Symbol.("x", 1:5))}(Tuple(X[:, j] for j in 1:5))
            problem = DecisionProblem(PayoffTable(Q), SampleBelief(draws; weights=w))
            exact = Float64(oracle_evpi(Q, p))
            e = evppi(problem, keys(draws))
            @test e.method === :GaussianProcessRegression && e.quantity === :evppi
            @test e.value ≈ exact rtol = 1e-3
            @test evpi(problem, GaussianProcessRegression()).value ≈ exact rtol = 1e-3

            # Five focal parameters on a 3⁵ grid and a sixth that is not learned: the
            # exact EVPPI is that of the partition into the 243 cells, which grouping
            # reproduces exactly and the process, which smooths, within a few per cent.
            levels = vec(collect(Iterators.product(fill(-1:1, 5)..., 1:3)))
            n = length(levels)
            G = [l[j] for l in levels, j in 1:5]
            h = [l[6] for l in levels]
            Q = zeros(BigRat, m, n)
            for a in 1:m
                b, c, pair = rand(rng, -3:3, 5), rand(rng, -2:2), rand(rng, 1:5, 2)
                d = rand(rng, -3:3)
                for i in 1:n
                    Q[a, i] = sum(b .* G[i, :]) + c * G[i, pair[1]] * G[i, pair[2]] +
                              d * (h[i] - 2) * (1 + G[i, 1])
                end
            end
            w = rand(rng, 1:4, n)
            p = BigRat.(w) .// sum(w)
            cell = [evalpoly(3, G[i, :] .+ 1) for i in 1:n]
            exact = oracle_evsi(Q, p, partition_likelihood(cell, BigRat))
            focal = Tuple(Symbol.("g", 1:5))
            draws = merge(NamedTuple{focal}(Tuple(G[:, j] for j in 1:5)), (; h))
            problem = DecisionProblem(PayoffTable(Q), SampleBelief(draws; weights=w))
            @test evppi(problem, focal, Grouping()).value == exact
            @test evppi(problem, focal).value ≈ Float64(exact) rtol = 0.05
        end
    end

    @testset "closed forms of the normal models" begin
        within(e, truth; slack=0.0) = abs(e.value - truth) < 4 * e.standard_error + slack
        # The R voi vignette's two-parameter model (helpers.jl).
        small = voi_normal_sample(Xoshiro(3), 4000)
        gp = GaussianProcessRegression()
        @test within(evppi(small, :p1, gp), voi_normal_value(1.0); slack=0.005)
        @test within(evppi(small, :p2, gp), voi_normal_value(2.0); slack=0.005)
        @test within(evppi(small, (:p1, :p2), gp), voi_normal_value(sqrt(5)); slack=0.005)
        @test within(evpi(small, gp), voi_normal_value(sqrt(5)); slack=0.005)

        # Six parameters: groups of five, and all six.
        problem = six_parameter_sample(Xoshiro(8), 4000)
        for indices in ([1, 3, 4, 5, 6], [2, 3, 4, 5, 6], [1, 2, 3, 4, 5], collect(1:6))
            e = evppi(problem, Tuple(Symbol.("x", indices)))
            @test e.method === :GaussianProcessRegression
            @test within(e, six_parameter_value(indices); slack=0.005)
            @test 0.005 < e.standard_error < 0.05
        end
        # The fit used a subsample (4000 draws, 1500 points) and predicted the rest;
        # conditioning on fewer points still estimates the same quantity.
        coarse = evppi(problem, Tuple(Symbol.("x", 1:5)),
                       GaussianProcessRegression(; max_points=400, hyper_points=150))
        @test within(coarse, six_parameter_value(1:5); slack=0.01)
        # Seeded: the same seed gives the same estimate, another seed another subsample.
        focal = Tuple(Symbol.("x", 1:5))
        again = evppi(problem, focal, GaussianProcessRegression(; max_points=400))
        @test again.value ==
              evppi(problem, focal,
                    GaussianProcessRegression(; max_points=400)).value
        @test again.value !=
              evppi(problem, focal,
                    GaussianProcessRegression(; max_points=400, seed=2)).value

        # Noise is not fitted: with the other parameters as noise, the estimate stays
        # well below the EVPI of the sample. (Length scales below the spacing of the
        # points would make the process pass for noise and reproduce the payoffs; the
        # lower bound on the length scales is what prevents that, and without it this
        # sample gives the EVPI back.)
        shared = Xoshiro(20261123)
        q1, q2 = 1 .+ randn(shared, 40_000), 2 .* randn(shared, 40_000)
        x = [randn(shared, 4000) for _ in 1:4]
        new = q1[1:4000] .- q2[1:4000] .+ 0.5 .* sum(x)
        noisy = DecisionProblem(PayoffTable(permutedims(hcat(zeros(4000), new))),
                                SampleBelief((; p1=q1[1:4000], p2=q2[1:4000], x3=x[1],
                                              x4=x[2], x5=x[3], x6=x[4])))
        guarded = evppi(noisy, (:p2, :x3, :x4, :x5, :x6))
        @test within(guarded, voi_normal_value(sqrt(5)); slack=0.005)
        @test guarded.value < evpi(noisy) - 2 * guarded.standard_error
        unguarded = evppi(noisy, (:p2, :x3, :x4, :x5, :x6),
                          GaussianProcessRegression(; min_length_scale=1e-3))
        # How far the unguarded fit overshoots depends on where the optimiser settles, which
        # differs between Julia versions and platforms, so only the direction is asserted.
        @test unguarded.value > guarded.value - guarded.standard_error

        # The contract of every regression: fitted rows average to the rows' means, and
        # a row that does not vary is fitted by its constant.
        Q = problem.payoff.Q
        w = problem.belief.weights
        columns = NamedTuple{focal}(Tuple(problem.belief.draws[name] for name in focal))
        fitted = fitted_conditional_means(GaussianProcessRegression(; max_points=400),
                                          columns, Q, w)
        @test size(fitted) == size(Q)
        @test fitted * Float64.(w) ≈ Q * Float64.(w)
        @test all(iszero, fitted[1, :])
        # The fitted means are close to the true conditional means 1 + Σ_{j ≤ 5} c_j x_j.
        truth = 1 .+ reduce(hcat, collect(columns)) * SIX_COEFFICIENTS[1:5]
        @test sqrt(sum(abs2, fitted[2, :] .- truth) / length(truth)) < 0.15

        # Weights: a draw of weight zero does not enter the fit or the estimate.
        n = 300
        base = six_parameter_sample(Xoshiro(9), n)
        weights = vcat(zeros(Int, 20), ones(Int, n - 20))
        weighted = DecisionProblem(base.payoff,
                                   SampleBelief(base.belief.draws; weights))
        dropped = DecisionProblem(PayoffTable(base.payoff.Q[:, 21:end];
                                              actions=[:reference, :new]),
                                  SampleBelief(map(c -> c[21:end], base.belief.draws)))
        @test evppi(weighted, focal).value ≈ evppi(dropped, focal).value rtol = 1e-8
    end
end
