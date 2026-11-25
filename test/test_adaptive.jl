# Sequential adaptive management (src/adaptive.jl; Lean: Finite/Adaptive.lean).

# Oracles written with normalised posteriors and no memory, which the package never
# uses: the value of the best policy and of the myopic (passive) policy over `n` periods
# from the belief `b`. `Ls[a]` is the likelihood of action `a`, or `nothing`.
function oracle_posteriors(b, L)
    L === nothing && return [(one(eltype(b)), b)]
    out = []
    for z in axes(L, 1)
        pz = sum(b[t] * L[z, t] for t in eachindex(b))
        iszero(pz) || push!(out, (pz, [b[t] * L[z, t] / pz for t in eachindex(b)]))
    end
    return out
end
function oracle_active(Q, Ls, discount, n, b)
    n == 0 && return zero(eltype(b)) * zero(eltype(Q))
    return maximum(axes(Q, 1)) do a
        later = sum(pz * oracle_active(Q, Ls, discount, n - 1, post)
                    for (pz, post) in oracle_posteriors(b, Ls[a]))
        return oracle_expected_utility(Q, b)[a] + discount * later
    end
end
function oracle_passive(Q, Ls, discount, n, b)
    n == 0 && return zero(eltype(b)) * zero(eltype(Q))
    eu = oracle_expected_utility(Q, b)
    a = findfirst(==(maximum(eu)), eu)
    later = sum(pz * oracle_passive(Q, Ls, discount, n - 1, post)
                for (pz, post) in oracle_posteriors(b, Ls[a]))
    return eu[a] + discount * later
end

@testset "adaptive management" begin
    # A safe action that pays nothing and shows nothing, and a risky one that pays 1
    # if the state is good and −3/2 if it is bad, and shows which.
    table = PayoffTable([:safe, :risky], [:bad, :good], [0 0; -3//2 1])
    problem = DecisionProblem(table, [1 // 2, 1 // 2]; baseline=:safe)
    shows = Experiment([1 0; 0 1]; signals=[:failed, :worked])

    @testset "probing, by hand" begin
        # The risky action loses 1/4 on the prior, so a manager who does not value
        # learning never tries it. Trying it once costs 1/4 and, half the time, earns 1
        # in each period that is left.
        for (horizon, value) in ((1, 0), (2, 1 // 4), (3, 3 // 4), (6, 9 // 4))
            r = adaptive_management(problem, (; risky=shows); horizon)
            @test r.active == value == max(0, -1 // 4 + (horizon - 1) // 2)
            @test r.no_learning == r.passive == 0
            @test r.perfect == horizon // 2
            @test r.value_of_probing == r.value_of_learning == value
            @test r.evpi == horizon // 2
            @test r.baseline == 0
            @test r.act_now_action == :safe
            @test r.first_action == (horizon == 1 ? :safe : :risky)
        end
        r = adaptive_management(problem, (; risky=shows); horizon=3)
        @test r.policy.period == 1 && r.policy.probability == 1
        @test r.policy.belief == [1 // 2, 1 // 2] && r.policy.value == 3 // 4
        failed, worked = last.(r.policy.branches)
        @test first.(r.policy.branches) == [:failed, :worked]
        @test (failed.action, failed.value, failed.belief) == (:safe, 0, [1, 0])
        @test (worked.action, worked.value, worked.probability) == (:risky, 2, 1 // 2)
        @test isempty(failed.branches)                    # :safe shows nothing
        @test only(worked.branches).first == :worked      # the other signal cannot occur
        @test only(worked.branches).second.period == 3
        # The same experiments given as pairs and as a dictionary.
        @test adaptive_management(problem, [:risky => shows]; horizon=3).active == 3 // 4
        @test adaptive_management(problem, Dict(:risky => shows); horizon=3).active ==
              3 // 4
        # Depth of the tree.
        @test adaptive_management(problem, (; risky=shows); horizon=3,
                                  depth=0).policy === nothing
        shallow = adaptive_management(problem, (; risky=shows); horizon=3, depth=1)
        @test isempty(shallow.policy.branches) && shallow.policy.action == :risky
        # Discounting: −1/4 now, then 1/2 a period at δ and δ².
        d = adaptive_management(problem, (; risky=shows); horizon=3, discount=1 // 2)
        @test d.active == -1 // 4 + 1 // 2 * (1 // 2 + 1 // 4)
        @test d.perfect == 1 // 2 * (1 + 1 // 2 + 1 // 4)
        # With a heavy discount the future is not worth the probe.
        heavy = adaptive_management(problem, (; risky=shows); horizon=3,
                                    discount=1 // 5)
        @test heavy.active == 0 && heavy.first_action == :safe
        @test adaptive_management(problem, (; risky=shows); horizon=4,
                                  discount=0).active == 0
    end

    @testset "limits" begin
        rng = Random.Xoshiro(20261129)
        for trial in 1:40
            m, n, nz = rand(rng, 2:3), rand(rng, 2:3), rand(rng, 2:3)
            base = exact_problem(rng, m, n)
            p = probabilities(base.belief)
            Q = base.payoff.Q
            horizon = rand(rng, 1:4)
            V0 = maximum(oracle_expected_utility(Q, p))
            # Nothing is ever learned.
            blind = adaptive_management(base, Pair{Symbol,Experiment}[]; horizon)
            @test blind.active == blind.passive == blind.no_learning == horizon * V0
            @test blind.value_of_learning == 0
            # One period: no future to learn for.
            e = Experiment(random_stochastic(rng, BigRat, nz, n))
            one = adaptive_management(base, e; horizon=1)
            @test one.active == one.passive == one.no_learning == act_now_value(base)
            @test one.perfect - one.no_learning == evpi(base)
            # The state is shown after the first period whatever is done.
            perfect = Experiment(BigRat[i == j for i in 1:n, j in 1:n])
            told = adaptive_management(base, perfect; horizon)
            @test told.active == told.passive == V0 + (horizon - 1) * (V0 + evpi(base))
            # What is seen does not depend on what is done: nothing to probe for.
            same = adaptive_management(base, e; horizon)
            @test same.active == same.passive
            @test same.first_action == same.act_now_action
            # Two periods with one experiment: act, see it, act again.
            two = adaptive_management(base, e; horizon=2)
            @test two.active == V0 + value_with_information(base, e)
        end
    end

    @testset "against the definitions, and the order of the four values" begin
        rng = Random.Xoshiro(20261130)
        probes, learns = 0, 0
        for trial in 1:120
            m, n = rand(rng, 2:3), rand(rng, 2:3)
            base = exact_problem(rng, m, n)
            p = probabilities(base.belief)
            Q = base.payoff.Q
            horizon = rand(rng, 1:4)
            discount = rand(rng, (1, 1, 9 // 10, 1 // 2))
            Ls = [rand(rng) < 0.25 ? nothing :
                  random_stochastic(rng, BigRat, rand(rng, 2:3), n)
                  for _ in 1:m]
            given = [base.payoff.actions[a] => Experiment(Ls[a])
                     for a in 1:m if Ls[a] !== nothing]
            r = adaptive_management(base, given; horizon, discount)
            @test r.active == oracle_active(Q, Ls, discount, horizon, p)
            @test r.passive == oracle_passive(Q, Ls, discount, horizon, p)
            weight = sum(discount^k for k in 0:(horizon - 1))
            @test r.no_learning == weight * maximum(oracle_expected_utility(Q, p))
            @test r.perfect ==
                  weight * (maximum(oracle_expected_utility(Q, p)) +
                            oracle_evpi(Q, p))
            # Lean: horizonWeight_mul_actNowValue_le_policyValue,
            # policyValue_le_adaptiveValue, adaptiveValue_le_horizonWeight_mul_perfectValue.
            @test r.no_learning <= r.passive <= r.active <= r.perfect
            @test r.value_of_probing == r.active - r.passive
            @test r.value_of_learning == r.active - r.no_learning <= r.evpi
            @test r.baseline === nothing
            probes += r.value_of_probing > 0
            learns += r.passive > r.no_learning
            # The tree: probabilities of the branches of a node sum to its own, and its
            # value is this period's expected payoff plus the discounted branches.
            node = r.policy
            @test node.value == r.active && node.belief == p
            if !isempty(node.branches)
                @test sum(b.probability for (_, b) in node.branches) == node.probability
                a = findfirst(==(node.action), base.payoff.actions)
                @test node.value ==
                      oracle_expected_utility(Q, node.belief)[a] +
                      discount * sum(b.probability * b.value for (_, b) in node.branches)
            end
            # Floating point agrees.
            fl = adaptive_management(DecisionProblem(PayoffTable(base.payoff.actions,
                                                                 base.payoff.states,
                                                                 Float64.(Q)),
                                                     Float64.(p)),
                                     [a => Experiment(Float64.(e.likelihood))
                                      for (a, e) in given]; horizon,
                                     discount=Float64(discount))
            @test fl.active ≈ r.active atol = 1e-9
            @test fl.passive ≈ r.passive atol = 1e-9
        end
        @test probes > 0 && learns > 0
    end

    @testset "repeated trials stay cheap" begin
        # A two-armed trial over 30 periods: each arm succeeds or fails, and the beliefs
        # depend only on the counts, so the recursion visits few of the 2^30 histories.
        # The weights are products of thirty likelihoods, so the rationals are big.
        arms = PayoffTable([:old, :new], [:worse, :better],
                           BigRat[1//2 1//2; 3//10 7//10])
        trial = DecisionProblem(arms, BigRat[3 // 5, 2 // 5]; baseline=:old)
        outcome = Experiment(BigRat[7//10 3//10; 3//10 7//10];
                             signals=[:failure, :success])
        r = adaptive_management(trial, (; new=outcome); horizon=30)
        @test r.beliefs_evaluated == 4960    # (periods left, successes, failures)
        @test r.no_learning == 15 && r.act_now_action == :old
        @test r.no_learning == r.passive < r.active < r.perfect
        @test r.first_action == :new
        @test r.baseline == 15
        # In machine rationals the same weights overflow, and say so.
        small = DecisionProblem(PayoffTable([:old, :new], [:worse, :better],
                                            [1//2 1//2; 3//10 7//10]), [3 // 5, 2 // 5])
        @test_throws OverflowError adaptive_management(small,
                                                       (;
                                                        new=Experiment([7//10 3//10;
                                                                        3//10 7//10])),
                                                       horizon=30)
    end

    @testset "feasibility" begin
        three = PayoffTable([:safe, :risky, :banned], [:bad, :good], [0 0; -3//2 1; 5 5])
        restricted = DecisionProblem(three, [1 // 2, 1 // 2]; baseline=:safe,
                                     feasible=[:safe, :risky])
        r = adaptive_management(restricted, (; risky=shows, banned=shows); horizon=3)
        @test r.active == 3 // 4 && r.perfect == 3 // 2
    end

    @testset "what is refused" begin
        @test_throws ArgumentError adaptive_management(problem, shows; horizon=0)
        @test_throws ArgumentError adaptive_management(problem, shows; horizon=2,
                                                       discount=-1)
        @test_throws ArgumentError adaptive_management(problem, shows; horizon=2,
                                                       depth=-1)
        @test_throws ArgumentError adaptive_management(problem, (; other=shows);
                                                       horizon=2)
        @test_throws ArgumentError adaptive_management(problem, (; risky=1); horizon=2)
        @test_throws LabelMismatchError adaptive_management(problem,
                                                            Experiment([1 0; 0 1];
                                                                       states=[:a, :b]);
                                                            horizon=2)
        credal = DecisionProblem(table, CredalSet([:bad, :good], [[0.5, 0.5], [0.4, 0.6]]))
        @test_throws BeliefFormError adaptive_management(credal, shows; horizon=2)
    end
end
