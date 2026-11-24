# Additions suggested by the Lean development: the separating decision problem of a
# refuted garbling, the prior that attains the upper EVPI, the maximality shortcut for
# E-admissibility, and the sharper constants of the two bounds. Oracles are explicit
# loops from the definitions (helpers.jl).

@testset "additions suggested by the Lean development" begin
    @testset "a refuted garbling carries a separating decision problem" begin
        rng = Xoshiro(61)
        refuted = 0
        for _ in 1:40
            n = rand(rng, 2:4)
            e = Experiment(random_stochastic(rng, BigRat, rand(rng, 2:3), n))
            f = Experiment(random_stochastic(rng, BigRat, rand(rng, 2:3), n))
            r = garbling_result(f, e)
            quick = garbling_result(f, e; separating=false)
            @test quick.is_garbling == r.is_garbling && quick.exact == r.exact
            @test quick.separating === nothing
            if r.is_garbling
                @test r.separating === nothing
                continue
            end
            refuted += 1
            p = r.separating
            @test p isa DecisionProblem && p.belief isa FiniteBelief
            @test size(p.payoff.Q) == (size(f.likelihood, 1), n)
            @test all(==(1 // n), p.belief.probabilities)
            # In that problem f is worth strictly more than e, so f cannot be a
            # garbling of e (Blackwell): checked from the definition.
            Q, prior = p.payoff.Q, p.belief.probabilities
            @test oracle_evsi(Q, prior, f.likelihood) > oracle_evsi(Q, prior, e.likelihood)
            @test evsi(p, f) > evsi(p, e)
            @test occursin("separating", sprint(show, r))
        end
        @test refuted >= 10
        # A garbling has none, and labelled states carry over.
        e = Experiment([4//5 1//5; 1//5 4//5]; states=[:dry, :wet], signals=[:low, :high])
        noisy = garble(e, [3//4 1//4; 1//4 3//4])
        @test garbling_result(noisy, e).separating === nothing
        back = garbling_result(e, noisy)
        @test !back.is_garbling && back.exact
        @test state_labels(back.separating) == [:dry, :wet]
        @test action_labels(back.separating.payoff) == [:act_on_low, :act_on_high]
        # Floating-point likelihoods: the answer is numerical and carries no problem.
        fe = Experiment([0.8 0.2; 0.2 0.8])
        fn = Experiment([0.65 0.35; 0.35 0.65])
        @test garbling_result(fe, fn).separating === nothing
    end

    @testset "the prior that attains the upper EVPI" begin
        rng = Xoshiro(62)
        for _ in 1:40
            m, n, k = rand(rng, 2:4), rand(rng, 2:4), rand(rng, 1:4)
            t = random_payoff_table(rng, m, n; T=BigRat)
            credal = random_credal_set(rng, n, k; T=BigRat)
            v = robust_evpi(DecisionProblem(t, credal))
            q = v.upper_prior
            @test length(q) == n && all(>=(0), q) && sum(q) == 1
            # The EVPI under that prior, from the definition, is the upper end.
            @test oracle_evpi(t.Q, q) == v.upper
            @test oracle_evpi(t.Q, q) >= maximum(v.members)
            # It lies in the hull: it never beats the hull's extreme expectations of
            # any state indicator.
            for j in 1:n
                column = [point[j] for point in credal.points]
                @test minimum(column) <= q[j] <= maximum(column)
            end
        end
        # The example of the docstring of `robust_evpi`: the upper end is interior.
        payoff = PayoffTable([4 0; 0 4; 3 3])
        credal = CredalSet([[9 // 10, 1 // 10], [1 // 10, 9 // 10]])
        v = robust_evpi(DecisionProblem(payoff, credal))
        @test oracle_evpi(payoff.Q, v.upper_prior) == v.upper > maximum(v.members)
        @test !(v.upper_prior in credal.points)
        @test occursin("upper end at the prior", sprint(show, v))
        # A single prior is its own upper prior; EVSI has no envelope.
        single = DecisionProblem(payoff, [1 // 3, 2 // 3])
        @test robust_evpi(single).upper_prior == [1 // 3, 2 // 3]
        e = Experiment([4//5 1//5; 1//5 4//5])
        @test robust_evsi(DecisionProblem(payoff, credal), e).upper_prior === nothing
    end

    @testset "not maximal implies not E-admissible, without a programme" begin
        rng = Xoshiro(63)
        skipped = 0
        for trial in 1:60
            m, n, k = rand(rng, 3:6), rand(rng, 2:4), rand(rng, 1:3)
            T = isodd(trial) ? BigRat : Float64
            t = random_payoff_table(rng, m, n; T)
            problem = DecisionProblem(t, random_credal_set(rng, n, k; T))
            result = decide(problem, EAdmissible())
            maximal = decide(problem, Maximal()).chosen
            actions = feasible_actions(problem)
            dominated = result.details.dominated_by
            @test sort(collect(keys(dominated))) == sort(setdiff(actions, maximal))
            skipped += length(dominated)
            E, _ = DecisionAnalysis._expectation_matrix(problem, EAdmissible())
            tol = T === BigRat ? 0 : sqrt(eps()) * maximum(abs, E)
            for (a, action) in enumerate(actions)
                # The shortcut agrees with the programme, solved for every action.
                admitted, = DecisionAnalysis._eadmissible_witness(E, a; tol)
                @test admitted == (action in result.chosen)
                haskey(dominated, action) || continue
                @test !admitted
                # The named action beats it at every generating point (explicit loop).
                b = findfirst(==(dominated[action]), actions)
                for point in problem.belief.points
                    eu(i) = sum(point[j] * t.Q[i, j] for j in 1:n)
                    @test eu(b) > eu(a)
                end
                T === BigRat && @test result.details.exact[action]
            end
            @test issubset(result.chosen, maximal)
        end
        @test skipped >= 20
    end

    @testset "sharper constants of the two bounds" begin
        rng = Xoshiro(64)
        for _ in 1:40
            m, n = rand(rng, 2:5), rand(rng, 2:5)
            t = random_payoff_table(rng, m, n; T=BigRat)
            Q = t.Q
            lo, hi = minimum(Q), maximum(Q)
            widest = maximum(maximum(Q[a, j] for j in 1:n) - minimum(Q[a, j] for j in 1:n)
                             for a in 1:m)
            @test payoff_range(t) == hi - lo
            @test largest_action_range(t) == widest <= payoff_range(t)
            @test largest_regret(t) == maximum(oracle_regret(Q)) <= payoff_range(t)

            # The decision-loss bound with the sharper constant holds, and is no
            # larger than the plain one.
            p = random_belief(rng, n; T=BigRat).probabilities
            q = random_belief(rng, n; T=BigRat).probabilities
            ε = total_variation(p, q)
            under_p = oracle_expected_utility(Q, p)
            chosen = first(argmax_set(oracle_expected_utility(Q, q)))
            loss = maximum(under_p) - under_p[chosen]
            problem = DecisionProblem(t, p)
            @test loss <= decision_loss_bound(problem, ε) == 2 * widest * ε
            @test decision_loss_bound(t, ε) <= decision_loss_bound(payoff_range(t), ε)

            # The value gap between two experiments is at most the largest regret
            # times the deficiency, which is at most the range times the deficiency.
            e = Experiment(random_stochastic(rng, BigRat, rand(rng, 2:3), n))
            f = Experiment(random_stochastic(rng, BigRat, rand(rng, 2:3), n))
            δ = deficiency(e, f)
            gap = oracle_evsi(Q, p, f.likelihood) - oracle_evsi(Q, p, e.likelihood)
            @test gap <= deficiency_value_bound(problem, δ) == largest_regret(t) * δ
            @test deficiency_value_bound(t, δ) <= payoff_range(t) * δ

            # Only the feasible actions of a problem count.
            m >= 3 || continue
            restricted = DecisionProblem(t, p; feasible=[true; true; falses(m - 2)])
            sub = PayoffTable(Q[1:2, :])
            @test payoff_range(restricted) == payoff_range(sub)
            @test largest_action_range(restricted) == largest_action_range(sub)
            @test largest_regret(restricted) == largest_regret(sub)
        end
        @test_throws ArgumentError deficiency_value_bound(PayoffTable([1 2; 3 4]), -1)
        @test_throws ArgumentError decision_loss_bound(PayoffTable([1 2; 3 4]), -1)
    end
end
