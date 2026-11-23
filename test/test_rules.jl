@testset "rules" begin
    Q = [0 -10 -60 -150; -20 -20 -25 -90; -48 -48 -48 -48]
    t = PayoffTable([:nothing, :levee, :relocate], [:low, :moderate, :high, :extreme], Q)
    p = [1 // 2, 3 // 10, 3 // 20, 1 // 20]
    problem = DecisionProblem(t, p)

    @testset "named rules are built from the combinators" begin
        @test MaxExpectedUtility() === DecisionRule(Expectation(); name=:MaxExpectedUtility)
        @test Maximin().metric === maximin_metric()
        @test Maximax().metric === maximax_metric()
        @test MinimaxRegret().metric === minimax_regret_metric()
        @test GammaMaximin().metric === RobustnessMetric(LowerExpectation())
        @test Satisfice(3).selector === Threshold(3)
        @test hurwicz_metric(0.3) === RobustnessMetric(Hurwicz(0.3))
        @test laplace_metric() === RobustnessMetric(Laplace())
        for rule in (MaxExpectedUtility(), Maximin(), Maximax(), MinimaxRegret())
            @test decide(problem, rule).scores == robustness(problem, rule.metric)
        end
    end

    @testset "worked values" begin
        r = decide(problem, MaxExpectedUtility())
        @test r.scores == [-39 // 2, -97 // 4, -48]
        @test r.chosen == [:nothing] && first_choice(r) == :nothing && r.sense === :max
        @test r.actions == t.actions
        @test decide(problem, Maximin()).chosen == [:relocate]
        @test decide(problem, Maximax()).chosen == [:nothing]
        r = decide(problem, MinimaxRegret())
        @test r.scores == [102, 42, 48] == max_regret(problem)
        @test r.chosen == [:levee] && r.sense === :min
        @test decide(problem, DecisionRule(Laplace())).chosen == [:levee]
        @test decide(problem, DecisionRule(Hurwicz(1 // 4))).chosen == [:relocate]
        @test decide(problem, DecisionRule(Hurwicz(9 // 10))).chosen == [:nothing]
        # Minimising expected regret is maximising expected utility.
        r = decide(problem, DecisionRule(Expectation(); transform=RegretTransform()))
        @test r.scores == expected_regret(problem) && r.chosen == [:nothing]
    end

    @testset "regret" begin
        @test regret_table(t) == [0 0 35 102; 20 10 0 42; 48 38 23 0]
        @test regret_table(problem) == regret_table(t)
        @test expected_regret(problem) == [207 // 20, 151 // 10, 777 // 20]
        rng = Random.Xoshiro(5)
        for _ in 1:30
            prob = exact_problem(rng, rand(rng, 1:5), rand(rng, 1:5))
            R = regret_table(prob)
            @test R == oracle_regret(prob.payoff.Q)
            @test all(>=(0), R)
            @test all(iszero, minimum(R; dims=1))
            @test expected_regret(prob) ==
                  oracle_expected_utility(R, probabilities(prob.belief))
            @test max_regret(prob) == [maximum(R[a, :]) for a in axes(R, 1)]
            # Regret ignores a state-dependent shift of all actions.
            shift = BigRat.(rand(rng, -9:9, 1, size(R, 2)))
            @test regret_table(PayoffTable(prob.payoff.Q .+ shift)) == R
        end
        # Regret over a scenario set is measured within the scenarios.
        scen = DecisionProblem(t, ScenarioSet([:low, :moderate]))
        @test regret_table(scen) == [0 0; 20 10; 48 38]
        @test max_regret(scen) == [0, 20, 48]
        @test_throws BeliefFormError expected_regret(scen)
        # Relative regret: positive payoffs, by hand.
        pos = PayoffTable([10 4; 5 8])
        @test relative_regret_table(pos) == [0 1//2; 1//2 0]
        @test relative_regret_table(DecisionProblem(pos, [1 // 2, 1 // 2])) ==
              relative_regret_table(pos)
        rr = DecisionRule(WorstCase(); transform=RelativeRegretTransform())
        @test decide(DecisionProblem(pos, ScenarioSet([:s1, :s2])), rr).scores ==
              [1 // 2, 1 // 2]
        # A zero best payoff has no relative regret.
        @test_throws ArgumentError relative_regret_table(t)
    end

    @testset "ties: all returned, first in table order chosen" begin
        tied = DecisionProblem(PayoffTable([:a, :b, :c], [:s, :t], [1 3; 3 1; 2 2]),
                               [1 // 2, 1 // 2])
        r = decide(tied, MaxExpectedUtility())
        @test r.chosen == [:a, :b, :c]
        @test first_choice(r) == :a
        # Reordering the table reorders the tie, deterministically.
        flipped = DecisionProblem(PayoffTable([:c, :b, :a], [:s, :t], [2 2; 3 1; 1 3]),
                                  [1 // 2, 1 // 2])
        @test first_choice(decide(flipped, MaxExpectedUtility())) == :c
        # Floating point: equal up to rounding is a tie; a real gap is not.
        f = DecisionProblem(PayoffTable([0.1+0.2 0.0; 0.3 0.0; 0.29 0.0]), [1.0, 0.0])
        @test decide(f, MaxExpectedUtility()).chosen == [:a1, :a2]
        @test decide(f, MaxExpectedUtility(); rtol=0).chosen == [:a1]
        @test decide(f, MaxExpectedUtility(); atol=0.02).chosen == [:a1, :a2, :a3]
        # Exact arithmetic has no tolerance at all.
        e = DecisionProblem(PayoffTable([1//1 0; 1 // 1-1 // 10^12 0]), [1 // 1, 0 // 1])
        @test decide(e, MaxExpectedUtility()).chosen == [:a1]
    end

    @testset "satisficing" begin
        r = decide(problem, Satisfice(-25))
        @test r.chosen == [:nothing, :levee] && first_choice(r) == :nothing
        @test decide(problem, Satisfice(-50; aggregator=WorstCase())).chosen == [:relocate]
        none = decide(problem, Satisfice(0; aggregator=WorstCase()))
        @test isempty(none.chosen) && first_choice(none) === nothing
        # On regret the threshold is a ceiling.
        r = decide(problem,
                   Satisfice(45; aggregator=WorstCase(), transform=RegretTransform()))
        @test r.chosen == [:levee] && r.sense === :min
    end

    @testset "minimax regret violates independence of irrelevant alternatives" begin
        states = ScenarioSet([:s1, :s2])
        two = DecisionProblem(PayoffTable([:A, :B], [:s1, :s2], [0 10; 6 3]), states)
        three = DecisionProblem(PayoffTable([:A, :B, :C], [:s1, :s2], [0 10; 6 3; 12 0]),
                                states)
        @test decide(two, MinimaxRegret()).scores == [6, 7]
        @test decide(two, MinimaxRegret()).chosen == [:A]
        @test decide(three, MinimaxRegret()).scores == [12, 7, 10]
        # C is not chosen, yet adding it moves the choice from A to B.
        @test decide(three, MinimaxRegret()).chosen == [:B]
        # Maximin and expected utility do not have this defect on the same tables.
        @test decide(two, Maximin()).chosen == [:B]
        @test :A ∉ decide(three, Maximin()).chosen
        filtered = DecisionProblem(three.payoff, states; feasible=[:A, :B])
        @test decide(filtered, MinimaxRegret()).chosen == [:A]
    end

    @testset "robustness metrics" begin
        @test robustness(problem, maximin_metric()) == [-150, -90, -48]
        @test robustness(problem, maximax_metric()) == [0, -20, -48]
        @test robustness(problem, laplace_metric()) == [-55, -155 // 4, -48]
        @test robustness(problem, hurwicz_metric(1 // 4)) == [-225 // 2, -145 // 2, -48]
        @test robustness(problem, minimax_regret_metric()) == [102, 42, 48]
        # A scenario subset: regret is measured within it.
        sub = minimax_regret_metric(; scenarios=[:low, :moderate])
        @test robustness(problem, sub) == [0, 20, 48]
        @test robustness(problem, sub) ==
              max_regret(restrict_scenarios(problem, [:low, :moderate]))
        rule = DecisionRule(sub; name=:MildRegret)
        @test decide(problem, rule).chosen == [:nothing]
        @test decide(problem,
                     DecisionRule(WorstCase(); scenarios=[:low, :moderate])).chosen ==
              [:nothing]
        # A probabilistic summary over a subset conditions the prior on it.
        cond = RobustnessMetric(Expectation(); scenarios=[:high, :extreme])
        @test robustness(problem, cond) == [-165 // 2, -165 // 4, -48]
        # Mean regret, a metric with no named rule.
        @test robustness(problem,
                         RobustnessMetric(Laplace(); transform=RegretTransform())) ==
              [137 // 4, 18, 109 // 4]
    end

    @testset "Pareto front" begin
        actions = [:a, :b, :c, :d]
        cost = DecisionProblem(PayoffTable(actions, [:s], reshape([-1, -2, -3, -3], 4, 1)),
                               [1 // 1])
        benefit = DecisionProblem(PayoffTable(actions, [:s], reshape([1, 5, 4, 9], 4, 1)),
                                  [1 // 1])
        r = decide([cost, benefit], ParetoFront())
        @test r.scores == [-1 1; -2 5; -3 4; -3 9]
        @test r.chosen == [:a, :b, :d]            # c is dominated by b
        @test decide([cost], ParetoFront()).chosen == [:a]
        @test decide([cost, benefit], ParetoFront(WorstCase())).chosen == [:a, :b, :d]
        # Equal rows do not dominate each other.
        same = DecisionProblem(PayoffTable([:x, :y], [:s], reshape([1, 1], 2, 1)), [1 // 1])
        @test decide([same, same], ParetoFront()).chosen == [:x, :y]
        other = DecisionProblem(PayoffTable([:a, :b], [:s], reshape([1, 2], 2, 1)),
                                [1 // 1])
        @test_throws LabelMismatchError decide([cost, other], ParetoFront())
        @test_throws ArgumentError decide(typeof(cost)[], ParetoFront())
        # Against brute force.
        rng = Random.Xoshiro(3)
        for _ in 1:20
            m, k = rand(rng, 1:6), rand(rng, 1:3)
            q = random_belief(rng, 3; T=BigRat)
            problems = [DecisionProblem(random_payoff_table(rng, m, 3; T=BigRat), q)
                        for _ in 1:k]
            S = reduce(hcat, [aggregate(Expectation(), pr) for pr in problems])
            keep = [a
                    for a in 1:m
                    if !any(b -> all(S[b, :] .>= S[a, :]) && any(S[b, :] .> S[a, :]), 1:m)]
            @test chosen_indices(decide(problems, ParetoFront())) == keep
        end
    end

    @testset "rules refuse beliefs they are not defined for" begin
        scen = DecisionProblem(t, ScenarioSet(t.states))
        @test_throws BeliefFormError decide(scen, MaxExpectedUtility())
        @test_throws BeliefFormError decide(scen, GammaMaximin())
        @test_throws BeliefFormError decide(scen, EAdmissible())
        @test_throws BeliefFormError decide(scen, Maximal())
        @test_throws BeliefFormError decide(scen, IntervalDominance())
        @test decide(scen, Maximin()).chosen == [:relocate]
        @test decide(scen, MinimaxRegret()).chosen == [:levee]
    end
end
