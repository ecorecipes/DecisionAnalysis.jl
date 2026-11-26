@testset "value of information" begin
    @testset "worked values" begin
        t = PayoffTable([:wait, :act], [:low, :high], [0 -100; -30 -30])
        problem = DecisionProblem(t, [3 // 5, 2 // 5])
        test = Experiment([4//5 1//5; 1//5 4//5]; signals=[:negative, :positive], cost=3)
        @test act_now_value(problem) == -30
        @test evpi(problem) == 18
        @test value_with_information(problem, test) == -106 // 5
        @test evsi(problem, test) == 44 // 5
        @test enbs(problem, test) == 29 // 5
        @test enbs(problem, test; population=10) == 85
        @test_throws ArgumentError enbs(problem, test; population=-1)
        @test evppi(problem, [:all, :all]) == 0
        @test evppi(problem, [1, 2]) == 18
        @test_throws ArgumentError evppi(problem, [1, 2, 3])
        @test_throws InvalidExperimentError evsi(problem, Experiment([1 0 0; 0 1 1]))
        labelled = Experiment([4//5 1//5; 1//5 4//5]; states=[:calm, :storm])
        @test_throws LabelMismatchError evsi(problem, labelled)
        # A known state leaves nothing to learn.
        known = DecisionProblem(t, PointBelief(:high))
        @test act_now_value(known) == -30
        @test evpi(known) == 0 && evsi(known, test) == 0
    end

    @testset "beliefs without a single prior are refused" begin
        t = PayoffTable([0 -100; -30 -30])
        e = Experiment([4//5 1//5; 1//5 4//5])
        for belief in (CredalSet([[3 // 5, 2 // 5], [1 // 2, 1 // 2]]),
                       ScenarioSet([:s1, :s2]))
            problem = DecisionProblem(t, belief)
            @test_throws BeliefFormError evpi(problem)
            @test_throws BeliefFormError evsi(problem, e)
            @test_throws BeliefFormError evppi(problem, [1, 2])
            @test_throws BeliefFormError act_now_value(problem)
            @test_throws BeliefFormError enbs(problem, e)
        end
    end

    @testset "signals of probability zero contribute nothing" begin
        t = PayoffTable([0 -100 5; -30 -30 7] .// 1)
        # The third signal is sent only by the third state, which has probability zero.
        e = Experiment([4//5 1//5 0; 1//5 4//5 0; 0 0 1])
        problem = DecisionProblem(t, [3 // 5, 2 // 5, 0 // 1])
        @test bayes_invert(problem.belief, e).posteriors[3] === nothing
        @test evsi(problem, e) == 44 // 5
        @test evsi(problem, e) == oracle_evsi(t.Q, [3 // 5, 2 // 5, 0 // 1], e.likelihood)
        f = DecisionProblem(PayoffTable(Float64.(t.Q)), [0.6, 0.4, 0.0])
        fe = Experiment(Float64.(e.likelihood))
        @test isfinite(evsi(f, fe)) && evsi(f, fe) ≈ 8.8
    end

    @testset "properties on random problems, exact" begin
        rng = Random.Xoshiro(20261123)
        for _ in 1:80
            m, n, nz = rand(rng, 1:5), rand(rng, 1:5), rand(rng, 1:4)
            problem = exact_problem(rng, m, n)
            Q, p = problem.payoff.Q, probabilities(problem.belief)
            e = random_experiment(rng, nz, n; T=BigRat)
            L = e.likelihood
            eu = aggregate(Expectation(), problem)
            er = expected_regret(problem)

            # Fast paths against the oracles.
            @test evpi(problem) == oracle_evpi(Q, p)
            @test value_with_information(problem, e) ==
                  oracle_value_with_information(Q, p, L)
            @test evsi(problem, e) == oracle_evsi(Q, p, L)
            @test act_now_value(problem) == maximum(oracle_expected_utility(Q, p))

            # EVPI is nonnegative, is the least expected regret, and the minimisers of
            # expected regret are the maximisers of expected utility.
            @test evpi(problem) >= 0
            @test evpi(problem) == minimum(er)
            @test argmin_set(er) == argmax_set(eu)
            @test decide(problem,
                         DecisionRule(Expectation(); transform=RegretTransform())).chosen ==
                  decide(problem, MaxExpectedUtility()).chosen
            @test er == maximum(eu .+ er) .- eu      # regret and utility differ by a constant

            # 0 ≤ EVSI ≤ EVPI, with equality at the two extreme experiments.
            @test 0 <= evsi(problem, e) <= evpi(problem)
            identity = Experiment(BigRat[i == j for i in 1:n, j in 1:n])
            @test evsi(problem, identity) == evpi(problem)
            @test evsi(problem, Experiment(ones(BigRat, 1, n))) == 0

            # Garbling never raises EVSI.
            G = random_stochastic(rng, BigRat, rand(rng, 1:3), nz)
            @test evsi(problem, garble(e, G)) <= evsi(problem, e)

            # EVPPI: zero for one cell, EVPI for single states, monotone under
            # refinement, and equal to the EVSI of the experiment that reveals the cell.
            coarse = rand(rng, 1:2, n)
            fine = [(coarse[t], rand(rng, 1:2)) for t in 1:n]
            @test evppi(problem, fill(:one, n)) == 0
            @test evppi(problem, 1:n) == evpi(problem)
            @test 0 <= evppi(problem, coarse) <= evppi(problem, fine) <= evpi(problem)
            for part in (coarse, fine)
                reveal = Experiment(partition_likelihood(part, BigRat))
                @test evppi(problem, part) == evsi(problem, reveal)
            end

            # Positive affine maps keep the Bayes actions and scale EVPI, EVSI and
            # regret by the slope.
            c, d = rand(rng, 1:9) // rand(rng, 1:9), rand(rng, -9:9) // 1
            scaled = DecisionProblem(PayoffTable(c .* Q .+ d), p)
            @test chosen_indices(decide(scaled, MaxExpectedUtility())) == argmax_set(eu)
            @test evpi(scaled) == c * evpi(problem)
            @test evsi(scaled, e) == c * evsi(problem, e)
            @test regret_table(scaled) == c .* regret_table(problem)
            @test chosen_indices(decide(scaled, MinimaxRegret())) ==
                  chosen_indices(decide(problem, MinimaxRegret()))

            # The 2Bε bound: deciding under a wrong belief q loses at most 2 B TV(p, q).
            q = probabilities(random_belief(rng, n; T=BigRat))
            wrong = decide(DecisionProblem(problem.payoff, q), MaxExpectedUtility())
            loss = maximum(eu) - eu[first(chosen_indices(wrong))]
            B = maximum(Q) - minimum(Q)
            @test 0 <= loss <= decision_loss_bound(B, total_variation(p, q))

            # Floating point agrees with exact arithmetic.
            f = DecisionProblem(PayoffTable(Float64.(Q)), Float64.(p))
            fe = Experiment(Float64.(L))
            @test evpi(f) ≈ Float64(evpi(problem)) atol = 1e-9
            @test evsi(f, fe) ≈ Float64(evsi(problem, e)) atol = 1e-9
            @test evppi(f, coarse) ≈ Float64(evppi(problem, coarse)) atol = 1e-9
            @test expected_regret(f) ≈ Float64.(er)
        end
    end

    @testset "properties on random problems, floating point" begin
        rng = Random.Xoshiro(7)
        tol = 1e-10
        for _ in 1:60
            m, n, nz = rand(rng, 1:5), rand(rng, 1:5), rand(rng, 1:4)
            problem = DecisionProblem(random_payoff_table(rng, m, n), random_belief(rng, n))
            Q, p = problem.payoff.Q, probabilities(problem.belief)
            e = random_experiment(rng, nz, n)
            @test evpi(problem) ≈ oracle_evpi(Q, p) atol = tol
            @test evsi(problem, e) ≈ oracle_evsi(Q, p, e.likelihood) atol = tol
            @test evpi(problem) >= -tol
            @test evpi(problem) ≈ minimum(expected_regret(problem)) atol = tol
            @test argmin(expected_regret(problem)) ==
                  argmax(aggregate(Expectation(), problem))
            @test -tol <= evsi(problem, e) <= evpi(problem) + tol
            G = Float64.(random_stochastic(rng, BigRat, rand(rng, 1:3), nz))
            @test evsi(problem, garble(e, G)) <= evsi(problem, e) + tol
            coarse = rand(rng, 1:2, n)
            fine = [(coarse[t], rand(rng, 1:2)) for t in 1:n]
            @test -tol <= evppi(problem, coarse) <= evppi(problem, fine) + tol
            @test evppi(problem, fine) <= evpi(problem) + tol
        end
    end

    @testset "gain over a baseline" begin
        t = PayoffTable([:nothing, :levee, :relocate], [:low, :moderate, :high, :extreme],
                        [0 -10 -60 -150; -20 -20 -25 -90; -48 -48 -48 -48])
        p = [1 // 2, 3 // 10, 3 // 20, 1 // 20]
        problem = DecisionProblem(t, p; baseline=:levee)
        @test problem.baseline == :levee
        @test DecisionProblem(t, p; baseline="levee").baseline == :levee
        @test DecisionProblem(t, p).baseline === nothing
        g = gain_over_baseline(problem)
        @test g isa PayoffTable{Int}
        @test g.Q == [20 10 -35 -60; 0 0 0 0; -28 -28 -23 42]
        @test g.actions == t.actions && g.states == t.states
        @test expected_gain_over_baseline(problem) == [19 // 4, 0, -95 // 4]
        # Errors: a baseline that is not an action, and gains with no baseline.
        @test_throws InvalidBaselineError DecisionProblem(t, p; baseline=:swim)
        @test_throws InvalidBaselineError DecisionProblem(t, ScenarioSet(t.states);
                                                          baseline=:swim)
        @test_throws MissingBaselineError gain_over_baseline(DecisionProblem(t, p))
        @test_throws MissingBaselineError expected_gain_over_baseline(DecisionProblem(t, p))
        # The table of gains needs no probabilities; the expected gain does.
        scen = DecisionProblem(t, ScenarioSet(t.states); baseline=:nothing)
        @test gain_over_baseline(scen).Q == t.Q .- t.Q[1:1, :]
        # Without a single prior the gain is a range (lower, upper), not an error:
        # over the scenarios for a scenario set, over the priors for a credal set.
        @test expected_gain_over_baseline(scen) ==
              (lower=[0, -20, -48], upper=[0, 60, 102])
        credal = DecisionProblem(t, CredalSet(t.states, [p]); baseline=:nothing)
        one = expected_gain_over_baseline(DecisionProblem(t, p; baseline=:nothing))
        @test expected_gain_over_baseline(credal) == (lower=one, upper=one)
        # The baseline survives a scenario restriction and need not be feasible.
        @test restrict_scenarios(problem, [:low, :high]).baseline == :levee
        barred = DecisionProblem(t, p; baseline=:nothing, feasible=[:levee, :relocate])
        @test gain_over_baseline(barred).actions == [:levee, :relocate]
        @test gain_over_baseline(barred).Q == (t.Q .- t.Q[1:1, :])[2:3, :]
        @test expected_gain_over_baseline(barred) == [-19 // 4, -57 // 2]

        rng = Random.Xoshiro(31)
        for _ in 1:60
            m, n, nz = rand(rng, 1:5), rand(rng, 1:5), rand(rng, 1:4)
            payoff = random_payoff_table(rng, m, n; T=BigRat)
            prior = random_belief(rng, n; T=BigRat)
            base = rand(rng, 1:m)
            problem = DecisionProblem(payoff, prior; baseline=payoff.actions[base])
            e = random_experiment(rng, nz, n; T=BigRat)
            g = gain_over_baseline(problem)
            @test eltype(g) == BigRat
            # Against the definition, and zero in the baseline's own row.
            @test g.Q == [payoff.Q[a, t] - payoff.Q[base, t] for a in 1:m, t in 1:n]
            @test all(iszero, g.Q[base, :])
            eu = aggregate(Expectation(), problem)
            @test expected_gain_over_baseline(problem) == eu .- eu[base]
            @test iszero(expected_gain_over_baseline(problem)[base])
            # Setting a baseline changes nothing else.
            plain = DecisionProblem(payoff, prior)
            @test decide(problem, MaxExpectedUtility()).chosen ==
                  decide(plain, MaxExpectedUtility()).chosen
            @test evpi(problem) == evpi(plain)
            # Restating the problem in gains subtracts a state-dependent constant from
            # every action: Bayes actions, regret, EVPI, EVPPI and EVSI are unchanged.
            gains = DecisionProblem(g, prior; baseline=payoff.actions[base])
            @test decide(gains, MaxExpectedUtility()).chosen ==
                  decide(problem, MaxExpectedUtility()).chosen
            @test regret_table(gains) == regret_table(problem)
            @test expected_regret(gains) == expected_regret(problem)
            @test decide(gains, MinimaxRegret()).chosen ==
                  decide(problem, MinimaxRegret()).chosen
            @test evpi(gains) == evpi(problem)
            @test evsi(gains, e) == evsi(problem, e)
            part = rand(rng, 1:2, n)
            @test evppi(gains, part) == evppi(problem, part)
            @test all(iszero, gain_over_baseline(gains).Q .- g.Q)
            # The same in floating point, up to rounding.
            fproblem = DecisionProblem(PayoffTable(Float64.(payoff.Q)),
                                       Float64.(probabilities(prior)); baseline=:a1)
            fgains = DecisionProblem(gain_over_baseline(fproblem), fproblem.belief)
            fe = Experiment(Float64.(e.likelihood))
            @test evpi(fgains) ≈ evpi(fproblem) atol = 1e-9
            @test evsi(fgains, fe) ≈ evsi(fproblem, fe) atol = 1e-9
            @test regret_table(fgains) ≈ regret_table(fproblem) atol = 1e-9
        end
    end

    @testset "fidelity" begin
        @test total_variation([1 // 2, 1 // 2], [1 // 4, 3 // 4]) == 1 // 4
        @test total_variation([1.0, 0.0], [0.0, 1.0]) == 1.0
        @test total_variation([0.3, 0.7], [0.3, 0.7]) == 0.0
        a = FiniteBelief([:x, :y], [1 // 2, 1 // 2])
        b = FiniteBelief([:x, :y], [1 // 4, 3 // 4])
        @test total_variation(a, b) == 1 // 4
        @test_throws LabelMismatchError total_variation(a,
                                                        FiniteBelief([:x, :z],
                                                                     [1 // 2, 1 // 2]))
        @test_throws ArgumentError total_variation([1.0], [0.5, 0.5])
        # Half the ℓ¹ distance is the largest difference in the probability of an event.
        rng = Random.Xoshiro(17)
        for _ in 1:20
            n = rand(rng, 1:5)
            p = probabilities(random_belief(rng, n; T=BigRat))
            q = probabilities(random_belief(rng, n; T=BigRat))
            events = [[(mask >> (i - 1)) & 1 == 1 for i in 1:n] for mask in 0:(2^n - 1)]
            @test total_variation(p, q) ==
                  maximum(abs(sum(p[A]) - sum(q[A])) for A in events)
        end
        @test decision_loss_bound(150, 1 // 100) == 3
        @test decision_loss_bound(2.0, 0.25) == 1.0
        @test decision_loss_bound(5, 0) == 0
        @test_throws ArgumentError decision_loss_bound(-1, 0.1)
        @test_throws ArgumentError decision_loss_bound(1, -0.1)
    end

    @testset "robust value of information" begin
        @testset "EVPI envelope" begin
            # Both generators have EVPI 1/10; the midpoint has 1/2.
            payoff = PayoffTable([1 0; 0 1])
            credal = CredalSet([[9 // 10, 1 // 10], [1 // 10, 9 // 10]])
            v = robust_evpi(DecisionProblem(payoff, credal))
            @test v.quantity === :evpi
            @test v.members == [1 // 10, 1 // 10]
            @test v.lower == 1 // 10
            @test v.upper == 1 // 2
            # Γ-maximin: perfect information is worth 1 for sure; acting now is worth 1/10.
            @test v.act_now == 1 // 10 && v.informed == 1 && v.value == 9 // 10
            optimist = robust_evpi(DecisionProblem(payoff, credal);
                                   rule=DecisionRule(UpperExpectation()))
            @test optimist.value == 1 // 10
            @test (optimist.lower, optimist.upper) == (v.lower, v.upper)
        end

        @testset "random credal sets" begin
            rng = Random.Xoshiro(55)
            for _ in 1:30
                m, n, K = rand(rng, 1:4), rand(rng, 2:4), rand(rng, 1:3)
                payoff = random_payoff_table(rng, m, n; T=BigRat)
                credal = random_credal_set(rng, n, K; T=BigRat)
                problem = DecisionProblem(payoff, credal)
                pts = extreme_points(credal)
                v = robust_evpi(problem)
                members = [evpi(DecisionProblem(payoff, q)) for q in pts]
                @test v.members == members
                @test v.lower == minimum(members)
                @test v.upper >= maximum(members)
                # The upper end against a grid over the set: never below it, and no
                # prior on the grid beats it.
                grid = [evpi(DecisionProblem(payoff, sum(w[k] * pts[k] for k in 1:K)))
                        for w in simplex_grid(K, 20)]
                @test v.upper >= maximum(grid)
                @test v.lower <= minimum(grid)
                # Γ-maximin EVPI by hand, and it is nonnegative.
                best = [maximum(payoff.Q[:, t]) for t in 1:n]
                informed = minimum(sum(q .* best) for q in pts)
                act_now = maximum(aggregate(LowerExpectation(), problem))
                @test v.informed == informed && v.act_now == act_now
                @test v.value == informed - act_now >= 0
                # One generator: everything is the ordinary EVPI.
                single = robust_evpi(DecisionProblem(payoff, pts[1]))
                @test single.value == single.lower == single.upper == members[1]
                # Floating point.
                fv = robust_evpi(DecisionProblem(PayoffTable(Float64.(payoff.Q)),
                                                 CredalSet([Float64.(q) for q in pts])))
                @test fv.upper ≈ Float64(v.upper) atol = 1e-9
                @test fv.lower ≈ Float64(v.lower) atol = 1e-9
                @test fv.value ≈ Float64(v.value) atol = 1e-9
            end
        end

        @testset "robust EVSI can be negative and is not clamped" begin
            # Dilation. X is a fair coin under every prior; Y is unknown; the signal is
            # X xor Y. Before the signal the bet on X is favourable; after it the
            # posterior for X is anything between 0 and 1, and Γ-maximin abstains.
            states = [:x0y0, :x1y0, :x0y1, :x1y1]
            payoff = PayoffTable([:bet_one, :bet_zero, :abstain], states,
                                 [-1 6//5 -1 6//5; 6//5 -1 6//5 -1; 0 0 0 0])
            credal = CredalSet(states, [[1 // 2, 1 // 2, 0, 0], [0, 0, 1 // 2, 1 // 2]])
            signal = Experiment([1 0 0 1; 0 1 1 0]; signals=[:same, :different])
            problem = DecisionProblem(payoff, credal)
            v = robust_evsi(problem, signal)
            @test v.quantity === :evsi
            @test v.act_now == 1 // 10
            @test v.informed == 0
            @test v.value == -1 // 10
            @test v.value < 0
            # Each member, taken alone, values the signal positively.
            @test v.members == [evsi(DecisionProblem(payoff, q), signal)
                                for q in extreme_points(credal)]
            @test all(>(0), v.members)
            @test v.lower === nothing && v.upper === nothing
            # In floating point too.
            fv = robust_evsi(DecisionProblem(PayoffTable(payoff.actions, states,
                                                         Float64.(payoff.Q)),
                                             CredalSet(states,
                                                       [Float64.(q)
                                                        for q in extreme_points(credal)])),
                             Experiment(Float64.(signal.likelihood)))
            @test fv.value ≈ -0.1
            # A single prior: the robust value is the ordinary EVSI.
            rng = Random.Xoshiro(77)
            for _ in 1:20
                single = exact_problem(rng, rand(rng, 1:4), 3)
                e = random_experiment(rng, rand(rng, 1:3), 3; T=BigRat)
                @test robust_evsi(single, e).value == evsi(single, e)
            end
            # An uninformative signal is worth exactly nothing under any rule.
            blank = Experiment([1 1 1 1])
            @test robust_evsi(problem, blank).value == 0
            @test robust_evsi(problem, blank;
                              rule=DecisionRule(AlphaMaxmin(1 // 2))).value ==
                  0
        end

        @testset "unsupported rules and beliefs" begin
            payoff = PayoffTable([1 0; 0 1])
            credal = DecisionProblem(payoff, CredalSet([[1 // 2, 1 // 2]]))
            e = Experiment([1 0; 0 1])
            for rule in (MinimaxRegret(), Satisfice(0), EAdmissible(), Maximin())
                @test_throws ArgumentError robust_evpi(credal; rule)
                @test_throws ArgumentError robust_evsi(credal, e; rule)
            end
            scen = DecisionProblem(payoff, ScenarioSet([:s1, :s2]))
            @test_throws BeliefFormError robust_evpi(scen)
            @test_throws BeliefFormError robust_evsi(scen, e)
        end
    end
end

@testset "cells of a partition in any order and with any labels" begin
    rng = Random.Xoshiro(20261206)
    for trial in 1:50
        m, n = rand(rng, 2:4), rand(rng, 2:9)
        problem = exact_problem(rng, m, n)
        p, Q = probabilities(problem.belief), problem.payoff.Q
        partition = rand(rng, [:low, "mid", 3, (1, 2)], n)        # mixed labels, unsorted
        cells = unique(partition)
        slow = sum(maximum(sum(p[j] * Q[:, j] for j in 1:n if partition[j] == c))
                   for c in cells) - maximum(Q * p)
        @test evppi(problem, partition) == slow
        @test DecisionAnalysis._cell_columns(partition) ==
              [findall(==(c), partition) for c in cells]
        # One cell for each state is perfect information; one cell for all is none.
        @test evppi(problem, collect(1:n)) == evpi(problem)
        @test evppi(problem, fill(:all, n)) == 0
    end
end
