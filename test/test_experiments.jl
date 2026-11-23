@testset "experiments" begin
    gauge = Experiment([9//10 3//5 1//5 1//20; 1//10 2//5 4//5 19//20];
                       signals=[:no_warning, :warning])
    flip = [4//5 1//5; 1//5 4//5]

    @testset "construction" begin
        @test signal_labels(gauge) == [:no_warning, :warning]
        @test state_labels(gauge) === nothing
        @test gauge.cost == 0
        @test eltype(gauge) == Rational{Int}
        e = Experiment([0.8 0.2; 0.2 0.8]; states=["low", "high"], cost=3.0)
        @test signal_labels(e) == [:z1, :z2] && state_labels(e) == [:low, :high]
        @test e.cost == 3.0
        # Rows are signals and columns are states: a row-stochastic matrix is refused.
        @test_throws InvalidExperimentError Experiment([9//10 1//10; 3//5 2//5; 1//5 4//5])
        @test_throws InvalidExperimentError Experiment([0.5 0.5; 0.6 0.5])
        @test_throws InvalidExperimentError Experiment([1.5 0.5; -0.5 0.5])
        @test_throws InvalidExperimentError Experiment([0.5 0.5; 0.5 0.5]; signals=[:a])
        @test_throws InvalidExperimentError Experiment([0.5 0.5; 0.5 0.5]; signals=[:a, :a])
        @test_throws InvalidExperimentError Experiment([0.5 0.5; 0.5 0.5]; states=[:a])
        @test_throws InvalidExperimentError Experiment(zeros(0, 2))
        @test_throws ArgumentError Experiment([0.5 0.5; 0.5 0.5]; cost=Inf)
    end

    @testset "from a FiniteKernel" begin
        Θ = FiniteKernels.FiniteSpace(:State, [:low, :high])
        Z = FiniteKernels.FiniteSpace(:Test, [:negative, :positive, :unclear])
        # User layout (parent, child): rows = states here; the kernel is outputs-first.
        k = FiniteKernels.cpt(only(FiniteKernels.factors(Θ)),
                              only(FiniteKernels.factors(Z)),
                              [0.7 0.2 0.1; 0.1 0.6 0.3])
        e = Experiment(k; cost=2)
        @test signal_labels(e) == [:negative, :positive, :unclear]
        @test state_labels(e) == [:low, :high]
        @test e.likelihood == [0.7 0.1; 0.2 0.6; 0.1 0.3]
        for (i, z) in enumerate(signal_labels(e)), (j, s) in enumerate(state_labels(e))
            @test e.likelihood[i, j] == FiniteKernels.probability(k, z, s)
        end
        @test e.cost == 2
        @test_throws ArgumentError Experiment(FiniteKernels.uniform(Θ))
    end

    @testset "Bayesian inversion" begin
        prior = FiniteBelief([:low, :high], [3 // 5, 2 // 5])
        test = Experiment([4//5 1//5; 1//5 4//5]; signals=[:negative, :positive])
        inv = bayes_invert(prior, test)
        @test inv.marginal == [14 // 25, 11 // 25]
        @test probabilities(inv.posteriors[1]) == [6 // 7, 1 // 7]
        @test probabilities(inv.posteriors[2]) == [3 // 11, 8 // 11]
        @test inv.joint == [12//25 2//25; 3//25 8//25]
        @test inv.signals == [:negative, :positive] && inv.states == [:low, :high]
        @test bayes_invert([3 // 5, 2 // 5], test).marginal == inv.marginal
        # The posteriors average back to the prior.
        rng = Random.Xoshiro(4)
        for _ in 1:30
            n, nz = rand(rng, 1:5), rand(rng, 1:4)
            q = random_belief(rng, n; T=BigRat)
            e = random_experiment(rng, nz, n; T=BigRat)
            inv = bayes_invert(q, e)
            @test sum(inv.marginal) == 1
            @test sum(inv.marginal[z] .* probabilities(inv.posteriors[z]) for z in 1:nz) ==
                  probabilities(q)
            finv = bayes_invert(Float64.(probabilities(q)),
                                Experiment(Float64.(e.likelihood)))
            @test finv.marginal ≈ Float64.(inv.marginal)
            for z in 1:nz
                @test probabilities(finv.posteriors[z]) ≈
                      Float64.(probabilities(inv.posteriors[z]))
            end
        end
        # A signal of probability zero has no posterior: nothing, not NaN.
        impossible = Experiment([1//2 1; 1//2 0; 0 0]; signals=[:a, :b, :never])
        inv = bayes_invert(prior, impossible)
        @test inv.marginal[3] == 0 && inv.posteriors[3] === nothing
        @test inv.posteriors[1] isa FiniteBelief
        # A signal that only a zero-probability state can send.
        certain = FiniteBelief([:low, :high], [1.0, 0.0])
        inv = bayes_invert(certain, Experiment([1.0 0.3; 0.0 0.7]))
        @test inv.posteriors[2] === nothing
        @test !any(isnan, inv.marginal) && !any(isnan, inv.joint)
        @test probabilities(inv.posteriors[1]) == [1.0, 0.0]
        labelled = Experiment([4//5 1//5; 1//5 4//5]; states=[:calm, :storm])
        @test_throws LabelMismatchError bayes_invert(prior, labelled)
        @test_throws InvalidExperimentError bayes_invert(prior, gauge)
    end

    @testset "garbling" begin
        g = garble(gauge, flip)
        @test g.likelihood == [37//50 14//25 8//25 23//100; 13//50 11//25 17//25 77//100]
        @test g.likelihood == flip * gauge.likelihood
        @test garble(gauge, flip; signals=[:x, :y]).signals == [:x, :y]
        @test garble(Experiment(gauge.likelihood; cost=5), flip).cost == 5
        # Merging both signals into one is a garbling to the uninformative experiment.
        @test garble(gauge, [1 1]).likelihood == [1 1 1 1]
        @test_throws InvalidExperimentError garble(gauge, [4//5 1//5; 1//5 3//5])
        @test_throws InvalidExperimentError garble(gauge, [1 0 0; 0 1 1])
        # A row-stochastic G (the transpose convention) is refused.
        @test_throws InvalidExperimentError garble(gauge, [9//10 1//10; 3//10 7//10])
    end

    @testset "is_garbling_of and deficiency" begin
        g = garble(gauge, flip)
        G = is_garbling_of(g, gauge)
        @test G !== nothing
        @test G * gauge.likelihood == g.likelihood
        @test all(>=(0), G) && all(==(1), sum(G; dims=1))
        @test is_garbling_of(gauge, g) === nothing
        @test deficiency(gauge, g) == 0
        @test deficiency(g, gauge) > 0
        @test is_garbling_of(gauge, gauge) !== nothing

        # Two experiments built to be incomparable: one separates the first state from
        # the rest, the other the last state from the rest.
        first_state = Experiment([1 0 0; 0 1 1])
        last_state = Experiment([1 1 0; 0 0 1])
        @test is_garbling_of(first_state, last_state) === nothing
        @test is_garbling_of(last_state, first_state) === nothing
        @test deficiency(first_state, last_state) == 1 // 2
        @test deficiency(last_state, first_state) == 1 // 2
        # Both are garblings of the experiment that reveals the state, and the
        # uninformative experiment is a garbling of both.
        full = Experiment([1 0 0; 0 1 0; 0 0 1])
        none = Experiment([1 1 1])
        for e in (first_state, last_state)
            @test is_garbling_of(e, full) !== nothing
            @test is_garbling_of(none, e) !== nothing
            @test is_garbling_of(full, e) === nothing
            @test deficiency(e, full) > 0
        end
        @test_throws InvalidExperimentError deficiency(gauge, first_state)
        @test_throws LabelMismatchError deficiency(Experiment([1 0; 0 1]; states=[:a, :b]),
                                                   Experiment([1 0; 0 1]; states=[:a, :c]))

        rng = Random.Xoshiro(99)
        for _ in 1:25
            n, nz, nw = rand(rng, 2:4), rand(rng, 2:3), rand(rng, 1:3)
            e = random_experiment(rng, nz, n; T=BigRat)
            Grand = random_stochastic(rng, BigRat, nw, nz)
            garbled = garble(e, Grand)
            # A garbled experiment: a witness is found and deficiency is zero.
            W = is_garbling_of(garbled, e)
            @test W !== nothing
            @test W * e.likelihood == garbled.likelihood
            @test all(>=(0), W) && all(==(1), sum(W; dims=1))
            @test deficiency(e, garbled) == 0
            # The other direction: a witness exactly when the deficiency is zero.
            back = is_garbling_of(e, garbled)
            δ = deficiency(garbled, e)
            @test 0 <= δ <= 1
            @test (back === nothing) == (δ > 0)
            back === nothing || @test back * garbled.likelihood == e.likelihood
            # The programme's own minimiser attains the value it reports.
            sol = solve_program(deficiency_model(garbled, e))
            d, M = sol.objective, sol.values.G
            @test all(>=(0), M) && all(==(1), sum(M; dims=1))
            @test maximum(sum(abs, M * garbled.likelihood .- e.likelihood; dims=1)) == d
            @test δ == d / 2
            # Deficiency bounds the value gap by B δ for any problem and prior.
            problem = exact_problem(rng, rand(rng, 2:4), n)
            B = maximum(problem.payoff.Q) - minimum(problem.payoff.Q)
            @test evsi(problem, e) - evsi(problem, garbled) <= B * δ
            # Floating point: same verdicts and the same value up to rounding.
            fe = Experiment(Float64.(e.likelihood))
            fg = garble(fe, Float64.(Grand))
            @test is_garbling_of(fg, fe) !== nothing
            @test deficiency(fe, fg) <= 1e-12
            @test deficiency(fg, fe) ≈ Float64(δ) atol = 1e-9
        end
    end

    @testset "deficiency against a grid search" begin
        # Two signals to two signals: G has two free entries, so a grid over them
        # bounds the minimum from above and, being fine, closely.
        rng = Random.Xoshiro(123)
        steps = 60
        for _ in 1:8
            e = random_experiment(rng, 2, 3; T=BigRat)
            f = random_experiment(rng, 2, 3; T=BigRat)
            δ = deficiency(e, f)
            best = minimum(Iterators.product(0:steps, 0:steps)) do (i, j)
                a, b = i // steps, j // steps
                G = [a b; 1-a 1-b]
                return maximum(sum(abs, G * e.likelihood .- f.likelihood; dims=1)) / 2
            end
            @test δ <= best
            # Moving G by one grid step moves each column's distance by at most 2/steps.
            @test best - δ <= 2 // steps
        end
    end
end
