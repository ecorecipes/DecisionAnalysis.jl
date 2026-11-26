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
            @test r.plan == fill(:safe, horizon)
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

# A state that moves (Lean: Finite/AdaptiveDynamic.lean). Oracles on normalised beliefs,
# with no memory; the fixed plan is found by listing every sequence of actions.
function oracle_moved(b, L, T, signal_of)
    step(x) = T === nothing ? x : T * x
    L === nothing && return [(one(eltype(b)), step(b))]
    out = []
    for z in axes(L, 1)
        joint = signal_of === :next ? step(b) .* L[z, :] : step(b .* L[z, :])
        pz = sum(joint)
        iszero(pz) || push!(out, (pz, joint ./ pz))
    end
    return out
end
function oracle_moving_active(Q, Ls, Ts, signal_of, discount, n, b)
    n == 0 && return zero(eltype(b)) * zero(eltype(Q))
    return maximum(axes(Q, 1)) do a
        later = sum(pz * oracle_moving_active(Q, Ls, Ts, signal_of, discount, n - 1, post)
                    for (pz, post) in oracle_moved(b, Ls[a], Ts[a], signal_of))
        return oracle_expected_utility(Q, b)[a] + discount * later
    end
end
function oracle_plan_value(Q, Ts, discount, plan, b)
    total = zero(eltype(b)) * zero(eltype(Q))
    for (t, a) in enumerate(plan)
        total += discount^(t - 1) * oracle_expected_utility(Q, b)[a]
        b = Ts[a] === nothing ? b : Ts[a] * b
    end
    return total
end
# The best fixed plan over n periods: its value and its first action (the first in
# lexicographic order among the best).
function oracle_fixed_plan(Q, Ts, discount, n, b)
    plans = vec(collect(Iterators.product(ntuple(_ -> axes(Q, 1), n)...)))
    values = [oracle_plan_value(Q, Ts, discount, reverse(plan), b) for plan in plans]
    top = maximum(values)
    return top, minimum(last(plan) for (plan, v) in zip(plans, values) if v == top)
end
function oracle_replanning(Q, Ls, Ts, signal_of, discount, n, b)
    n == 0 && return zero(eltype(b)) * zero(eltype(Q))
    a = last(oracle_fixed_plan(Q, Ts, discount, n, b))
    later = sum(pz * oracle_replanning(Q, Ls, Ts, signal_of, discount, n - 1, post)
                for (pz, post) in oracle_moved(b, Ls[a], Ts[a], signal_of))
    return oracle_expected_utility(Q, b)[a] + discount * later
end
function oracle_state_seen(Q, Ts, discount, n, t)
    n == 0 && return zero(eltype(Q))
    return maximum(axes(Q, 1)) do a
        later = Ts[a] === nothing ? oracle_state_seen(Q, Ts, discount, n - 1, t) :
                sum(Ts[a][s, t] * oracle_state_seen(Q, Ts, discount, n - 1, s)
                    for s in axes(Q, 2))
        return Q[a, t] + discount * later
    end
end

@testset "adaptive management, a state that moves" begin
    @testset "by hand" begin
        # The state alternates for certain, and each action pays 1 in its own state.
        match = PayoffTable([:for_a, :for_b], [:a, :b], [1 0; 0 1])
        swap = [0 1; 1 0]
        nothing_seen = Pair{Symbol,Experiment}[]
        # Known at the start, so known for ever: the fixed plan alternates.
        known = DecisionProblem(match, [1, 0]; baseline=:for_a)
        for horizon in 1:5
            r = adaptive_management(known, nothing_seen; horizon, transitions=swap)
            @test r.no_learning == r.passive == r.active == r.perfect == horizon
            @test r.baseline == cld(horizon, 2)      # :for_a is right every other period
            @test r.act_now_action == :for_a
        end
        # Unknown at the start: without a signal, half of each period; told after the
        # first period, right from the second on.
        unknown = DecisionProblem(match, [1 // 2, 1 // 2])
        tells = Experiment([1 0; 0 1]; signals=[:saw_a, :saw_b])
        for signal_of in (:current, :next), horizon in 1:5
            blind = adaptive_management(unknown, nothing_seen; horizon, transitions=swap,
                                        signal_of)
            @test blind.no_learning == blind.passive == blind.active == horizon // 2
            @test blind.perfect == horizon
            told = adaptive_management(unknown, tells; horizon, transitions=swap,
                                       signal_of)
            @test told.no_learning == horizon // 2
            @test told.passive == told.active == 1 // 2 + (horizon - 1)
        end
        # The two conventions name the branch differently: after :saw_a of the current
        # state the next state is b; after :saw_a of the next state it is a.
        current = adaptive_management(unknown, tells; horizon=2, transitions=swap)
        next = adaptive_management(unknown, tells; horizon=2, transitions=swap,
                                   signal_of=:next)
        @test Dict(current.policy.branches)[:saw_a].belief == [0, 1]
        @test Dict(current.policy.branches)[:saw_a].action == :for_b
        @test Dict(next.policy.branches)[:saw_a].belief == [1, 0]
        @test Dict(next.policy.branches)[:saw_a].action == :for_a
    end

    @testset "planning ahead" begin
        # Sowing pays nothing now and makes the field ready; harvesting pays 3 from a
        # ready field and empties it. The best action for one period is never to sow.
        field = PayoffTable([:wait, :sow, :harvest], [:empty, :ready], [0 0; -1 -1; 0 3])
        problem = DecisionProblem(field, [1, 0]; baseline=:wait)
        moves = (; wait=[1 0; 0 1], sow=[0 0; 1 1], harvest=[1 1; 0 0])
        r = adaptive_management(problem, Pair{Symbol,Experiment}[]; horizon=4,
                                transitions=moves)
        @test r.no_learning == r.passive == r.active == r.perfect == 4   # sow, harvest ×2
        @test r.act_now_action == r.first_action == :sow
        @test r.plan == [:sow, :harvest, :sow, :harvest]
        @test r.baseline == 0
    end

    @testset "limits" begin
        rng = Random.Xoshiro(20261201)
        for trial in 1:40
            m, n, nz = rand(rng, 2:3), rand(rng, 2:3), rand(rng, 2:3)
            base = exact_problem(rng, m, n)
            p = probabilities(base.belief)
            Q = base.payoff.Q
            horizon = rand(rng, 1:4)
            e = Experiment(random_stochastic(rng, BigRat, nz, n))
            given = [base.payoff.actions[1] => e]
            signal_of = rand(rng, (:current, :next))
            # A state that stays where it is.
            still = adaptive_management(base, given; horizon,
                                        transitions=BigRat[i == j for i in 1:n, j in 1:n],
                                        signal_of)
            fixed = adaptive_management(base, given; horizon)
            for field in (:no_learning, :passive, :active, :perfect, :act_now_action,
                          :first_action)
                @test getfield(still, field) == getfield(fixed, field)
            end
            # A state drawn afresh each period, whatever it was: what is seen of this
            # period says nothing about the next.
            fresh = random_stochastic(rng, BigRat, n, 1)[:, 1]
            mixing = adaptive_management(base, given; horizon,
                                         transitions=repeat(fresh, 1, n))
            V0(q) = maximum(oracle_expected_utility(Q, q))
            @test mixing.active == mixing.passive == mixing.no_learning ==
                  V0(p) + (horizon - 1) * V0(fresh)
            @test mixing.perfect ==
                  V0(p) + oracle_evpi(Q, p) +
                  (horizon - 1) * (V0(fresh) + oracle_evpi(Q, fresh))
            # The next state shown after every period, whatever is done.
            T = random_stochastic(rng, BigRat, n, n)
            shown = adaptive_management(base,
                                        Experiment(BigRat[i == j
                                                          for i in 1:n, j in 1:n]);
                                        horizon, transitions=T, signal_of=:next)
            q, total = p, V0(p)
            for _ in 2:horizon
                q = T * q
                total += V0(q) + oracle_evpi(Q, q)
            end
            @test shown.active == shown.passive == total
        end
    end

    @testset "against the definitions, and the order of the four values" begin
        rng = Random.Xoshiro(20261202)
        probes, learns, replans = 0, 0, 0
        for trial in 1:150
            m, n = rand(rng, 2:3), rand(rng, 2:3)
            plain = exact_problem(rng, m, n)
            base = rand(rng, Bool) ? plain :
                   DecisionProblem(plain.payoff, plain.belief;
                                   baseline=plain.payoff.actions[rand(rng, 1:m)])
            p = probabilities(base.belief)
            Q = base.payoff.Q
            horizon = rand(rng, 1:4)
            discount = rand(rng, (1, 1, 9 // 10, 1 // 2))
            signal_of = rand(rng, (:current, :next))
            Ls = [rand(rng) < 0.25 ? nothing :
                  random_stochastic(rng, BigRat, rand(rng, 2:3), n)
                  for _ in 1:m]
            given = [base.payoff.actions[a] => Experiment(Ls[a])
                     for a in 1:m if Ls[a] !== nothing]
            shared = rand(rng, Bool)
            Ts = shared ? fill(random_stochastic(rng, BigRat, n, n), m) :
                 [random_stochastic(rng, BigRat, n, n) for _ in 1:m]
            transitions = shared ? Ts[1] : [base.payoff.actions[a] => Ts[a] for a in 1:m]
            r = adaptive_management(base, given; horizon, discount, transitions,
                                    signal_of)
            @test r.active ==
                  oracle_moving_active(Q, Ls, Ts, signal_of, discount, horizon, p)
            plan_value, plan_first = oracle_fixed_plan(Q, Ts, discount, horizon, p)
            @test r.no_learning == plan_value
            @test r.act_now_action == base.payoff.actions[plan_first] == first(r.plan)
            @test oracle_plan_value(Q, Ts, discount,
                                    indexin(r.plan, base.payoff.actions), p) == plan_value
            @test r.passive ==
                  oracle_replanning(Q, Ls, Ts, signal_of, discount, horizon, p)
            @test r.perfect ==
                  sum(p[t] * oracle_state_seen(Q, Ts, discount, horizon, t) for t in 1:n)
            # Lean: openLoopValue_le_dynamicPolicyValue, dynamicPolicyValue_le_dynamicValue,
            # dynamicValue_le_sum_observedValue.
            @test r.no_learning <= r.passive <= r.active <= r.perfect
            if base.baseline === nothing
                @test r.baseline === nothing
            else
                b = findfirst(==(base.baseline), base.payoff.actions)
                @test r.baseline ==
                      oracle_plan_value(Q, Ts, discount, fill(b, horizon), p)
                @test r.baseline <= r.no_learning
            end
            probes += r.active > r.passive
            learns += r.passive > r.no_learning
            # The fixed plan is not one action repeated.
            replans += r.no_learning >
                       maximum(oracle_plan_value(Q, Ts, discount, fill(a, horizon), p)
                               for a in 1:m)
            node = r.policy
            @test node.value == r.active && node.belief == p
            if !isempty(node.branches)
                @test sum(b.probability for (_, b) in node.branches) == node.probability
                @test all(sum(b.belief) == 1 for (_, b) in node.branches)
            end
            fl = adaptive_management(DecisionProblem(PayoffTable(base.payoff.actions,
                                                                 base.payoff.states,
                                                                 Float64.(Q)),
                                                     Float64.(p)),
                                     [a => Experiment(Float64.(e.likelihood))
                                      for (a, e) in given]; horizon,
                                     discount=Float64(discount),
                                     transitions=shared ? Float64.(Ts[1]) :
                                                 [a => Float64.(T)
                                                  for (a, T) in transitions], signal_of)
            @test fl.active ≈ r.active atol = 1e-9
            @test fl.passive ≈ r.passive atol = 1e-9
            @test fl.no_learning ≈ r.no_learning atol = 1e-9
            @test fl.perfect ≈ r.perfect atol = 1e-9
        end
        @test probes > 0 && learns > 0 && replans > 0
    end

    @testset "what is refused" begin
        table = PayoffTable([:safe, :risky], [:bad, :good], [0 0; -3//2 1])
        problem = DecisionProblem(table, [1 // 2, 1 // 2]; baseline=:safe)
        shows = Experiment([1 0; 0 1]; signals=[:failed, :worked])
        refused(; kwargs...) = @test_throws ArgumentError adaptive_management(problem,
                                                                              shows;
                                                                              horizon=2,
                                                                              kwargs...)
        refused(; transitions=[1 0 0; 0 1 0; 0 0 1])          # wrong size
        refused(; transitions=[1 1; 1 0])                     # a column sums to 2
        refused(; transitions=[1//2 1; 1//2 1])               # rows, not columns, sum to 1
        refused(; transitions=[3//2 0; -1//2 1])              # a negative entry
        refused(; transitions=(; risky=[1 0; 0 1]))           # :safe has no matrix
        refused(; transitions=(; safe=[1 0; 0 1], risky=[1 0; 0 1], other=[1 0; 0 1]))
        refused(; transitions=(; safe=[1 0; 0 1], risky=:stay))
        refused(; transitions=[1 0; 0 1], signal_of=:before)
        refused(; signal_of=:later)
    end
end

# The value function as a set of lines (method = :vectors; Lean:
# Finite/ValueVectors.lean), against the recursion over beliefs, which shares none of
# its code beyond the one-step look-ahead.
@testset "adaptive management by vectors" begin
    @testset "the upper envelope" begin
        prune(G) = (keep = DecisionAnalysis._envelope(G); (G[keep, :], keep))
        # A row is (value in state 1, value in state 2): a line from G[i, 1] at x = 0
        # to G[i, 2] at x = 1, where x is the probability of state 2.
        G = [4 0; 0 4; 3 3; 2 2; 1 1; 4 0; 3 1; 5//2 5//2]
        kept, tags = prune(G)
        @test tags == [1, 2, 3]                 # 4 and 5 lie below; 6 repeats 1;
        @test kept == G[[1, 2, 3], :]           # 7 lies under the envelope; 8 under 3
        # A line that only touches the envelope where two others cross is not needed.
        @test last(prune([4 0; 0 4; 2 2])) == [1, 2]
        # One that is the highest only at an end point is not needed either.
        @test last(prune([4 0; 4 -1])) == [1]
        @test last(prune([0 4; -1 4; 4 0])) == [1, 3]
        @test last(prune([1 1; 1 1])) == [1]
        @test last(prune(reshape([7, 9], 1, 2))) == [1]
        # The maximum is unchanged at every belief of a grid.
        rng = Random.Xoshiro(20261203)
        for trial in 1:200
            G = rand(rng, -5:5, rand(rng, 1:12), 2) .// 1
            kept, tags = prune(G)
            @test kept == G[tags, :] && issorted(tags) && allunique(tags)
            for x in 0:(1 // 24):1
                @test maximum(kept * [1 - x, x]) == maximum(G * [1 - x, x])
            end
            # Every line kept is the only maximum somewhere.
            for i in axes(kept, 1)
                @test any(x -> all(kept[i, :]' * [1 - x, x] > kept[j, :]' * [1 - x, x]
                                   for j in axes(kept, 1) if j != i),
                          0:(1 // 840):1)
            end
        end
    end

    @testset "against the recursion over beliefs" begin
        rng = Random.Xoshiro(20261204)
        many = 0
        for trial in 1:200
            m = rand(rng, 2:4)
            base = exact_problem(rng, m, 2)
            p = probabilities(base.belief)
            Q = base.payoff.Q
            horizon = rand(rng, 1:5)
            discount = rand(rng, (1, 1, 9 // 10, 1 // 2, 0))
            signal_of = rand(rng, (:current, :next))
            Ls = [rand(rng) < 0.25 ? nothing :
                  random_stochastic(rng, BigRat, rand(rng, 2:3), 2)
                  for _ in 1:m]
            given = [base.payoff.actions[a] => Experiment(Ls[a])
                     for a in 1:m if Ls[a] !== nothing]
            Ts = rand(rng) < 0.3 ? fill(nothing, m) :
                 [random_stochastic(rng, BigRat, 2, 2) for _ in 1:m]
            transitions = Ts[1] === nothing ? nothing :
                          [base.payoff.actions[a] => Ts[a] for a in 1:m]
            function run(; kwargs...)
                return adaptive_management(base, given; horizon, discount,
                                           transitions, signal_of, kwargs...)
            end
            slow, fast = run(), run(; method=:vectors)
            for field in (:no_learning, :passive, :active, :perfect, :value_of_learning,
                          :value_of_probing, :evpi, :baseline, :act_now_action,
                          :first_action, :plan, :policy)
                @test getfield(slow, field) == getfield(fast, field)
            end
            @test slow.value_function === nothing
            @test length(fast.value_function) == horizon
            many += size(fast.value_function[end].vectors, 1) > 2
            # The lines for n periods left give the value at any belief, and the action
            # of the highest line is a best first action there.
            for n in 1:horizon
                pieces = fast.value_function[n]
                q = random_stochastic(rng, BigRat, 2, 1)[:, 1]
                top, row = findmax(pieces.vectors * q)
                @test top ==
                      adaptive_management(DecisionProblem(base.payoff, q), given;
                                          horizon=n, discount, transitions, signal_of,
                                          depth=0, passive=false).active
                a = findfirst(==(pieces.actions[row]), base.payoff.actions)
                later = n == 1 ? 0 :
                        sum(pz * maximum(fast.value_function[n - 1].vectors * post)
                            for (pz, post) in oracle_moved(q, Ls[a], Ts[a], signal_of))
                @test oracle_expected_utility(Q, q)[a] + discount * later == top
            end
            # Without the passive manager.
            bare = run(; method=:vectors, passive=false)
            @test bare.passive === nothing && bare.value_of_probing === nothing
            @test bare.active == slow.active && bare.no_learning == slow.no_learning
            @test run(; passive=false).passive === nothing
            # Floating point.
            fl = adaptive_management(DecisionProblem(PayoffTable(base.payoff.actions,
                                                                 base.payoff.states,
                                                                 Float64.(Q)),
                                                     Float64.(p)),
                                     [a => Experiment(Float64.(e.likelihood))
                                      for (a, e) in given]; horizon,
                                     discount=Float64(discount),
                                     transitions=transitions === nothing ? nothing :
                                                 [a => Float64.(T)
                                                  for (a, T) in transitions], signal_of,
                                     method=:vectors)
            @test fl.active ≈ slow.active atol = 1e-9
            @test fl.passive ≈ slow.passive atol = 1e-9
            @test fl.no_learning ≈ slow.no_learning atol = 1e-9
        end
        @test many > 0
    end

    @testset "long horizons" begin
        # A reserve that a weed can invade (vignette 20), in exact arithmetic: 60 years,
        # far beyond what the recursion over beliefs can reach.
        table = PayoffTable([:none, :inspect, :treat], [:clear, :infested],
                            BigRat[50 20; 48 18; 40 10])
        untreated = BigRat[23//25 0; 2//25 1]
        treated = BigRat[23//25 391//500; 2//25 109//500]
        moves = (; none=untreated, inspect=untreated, treat=treated)
        noticed(d) = Experiment(BigRat[1 1-d; 0 d]; signals=[:not_seen, :seen])
        seen = (; none=noticed(1 // 5), inspect=noticed(9 // 10), treat=noticed(9 // 10))
        problem = DecisionProblem(table, BigRat[9 // 10, 1 // 10]; baseline=:none)
        far = adaptive_management(problem, seen; horizon=60, transitions=moves, depth=2,
                                  method=:vectors, passive=false)
        @test far.baseline < far.no_learning < far.active < far.perfect
        @test length(far.plan) == 60 && count(==(:treat), far.plan) > 5
        @test !(:inspect in far.plan)                 # a fixed plan never inspects
        @test far.first_action == :inspect
        @test size(far.value_function[60].vectors, 1) < 200
        # The lines for 8 years left are those of an 8-year problem.
        eight = adaptive_management(problem, seen; horizon=8, transitions=moves, depth=0)
        @test maximum(far.value_function[8].vectors * BigRat[9 // 10, 1 // 10]) ==
              eight.active
        # A fixed state with repeated trials is the case where the beliefs coincide and
        # the lines are as many as the beliefs: the two methods agree, and the
        # recursion over beliefs is the one to use.
        arms = PayoffTable([:old, :new], [:worse, :better],
                           BigRat[1//2 1//2; 3//10 7//10])
        trial = DecisionProblem(arms, BigRat[3 // 5, 2 // 5]; baseline=:old)
        outcome = Experiment(BigRat[7//10 3//10; 3//10 7//10];
                             signals=[:failure, :success])
        slow = adaptive_management(trial, (; new=outcome); horizon=16, depth=0)
        fast = adaptive_management(trial, (; new=outcome); horizon=16, depth=0,
                                   method=:vectors)
        @test slow.active == fast.active && slow.passive == fast.passive
        @test size(fast.value_function[16].vectors, 1) > 50
    end

    @testset "what is refused" begin
        three = DecisionProblem(PayoffTable([:a, :b], [:x, :y, :z], [1 0 0; 0 1 1]),
                                [1 // 3, 1 // 3, 1 // 3])
        e3 = Experiment([1 0 0; 0 1 1])
        @test_throws ArgumentError adaptive_management(three, e3; horizon=2,
                                                       method=:vectors)
        @test adaptive_management(three, e3; horizon=2).active == 5 // 3   # 2/3, then 1
        two = DecisionProblem(PayoffTable([:a, :b], [:x, :y], [1 0; 0 1]), [1 // 2, 1 // 2])
        @test_throws ArgumentError adaptive_management(two, Experiment([1 0; 0 1]);
                                                       horizon=2, method=:grid)
    end
end

# Bounds from a grid of beliefs (method = :points; Lean: Finite/ValueVectors.lean,
# sum_mul_backupVector_le_dynamicValue and dynamicValue_weighted_sum_le), against the
# exact recursion over beliefs.
@testset "adaptive management by points" begin
    @testset "the grid" begin
        grid = DecisionAnalysis._simplex_grid(Rational{Int}, 3, 4)
        @test length(grid) == binomial(4 + 2, 2) && allunique(grid)
        @test grid[1:3] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        @test all(sum(q) == 1 && all(>=(0), q) && all(isinteger, 4 .* q) for q in grid)
        @test length(DecisionAnalysis._simplex_grid(Float64, 4, 5)) == binomial(8, 3)
        @test DecisionAnalysis._simplex_grid(Rational{Int}, 2, 1) == [[1, 0], [0, 1]]
    end

    @testset "the exact value lies between the bounds" begin
        rng = Random.Xoshiro(20261205)
        tight, narrower = 0, 0
        for trial in 1:60
            m, n = rand(rng, 2:3), rand(rng, 2:4)
            base = exact_problem(rng, m, n)
            horizon = rand(rng, 1:4)
            discount = rand(rng, (1, 9 // 10, 1 // 2))
            signal_of = rand(rng, (:current, :next))
            given = [base.payoff.actions[a] => Experiment(random_stochastic(rng, BigRat,
                                                                            rand(rng, 2:3),
                                                                            n))
                     for a in 1:m if rand(rng) < 0.75]
            kind = rand(rng, 1:3)       # fixed state, one transition, one for each action
            transitions = kind == 1 ? nothing :
                          kind == 2 ? random_stochastic(rng, BigRat, n, n) :
                          [base.payoff.actions[a] => random_stochastic(rng, BigRat, n, n)
                           for a in 1:m]
            function run(; kwargs...)
                return adaptive_management(base, given; horizon, discount,
                                           transitions, signal_of, kwargs...)
            end
            exact = run()
            coarse = run(; method=:points, grid=3)
            fine = run(; method=:points, grid=9)
            for r in (coarse, fine)
                @test r.active <= exact.active <= r.active_upper
                @test r.perfect == exact.perfect && r.baseline == exact.baseline
                @test r.active_upper <= r.perfect
                @test r.passive === nothing && r.value_of_probing === nothing
                @test length(r.value_function) == horizon
                # The lower bound is the look-ahead on the vectors for one period fewer.
                @test r.policy.value == r.active
                if kind == 3
                    @test r.no_learning === nothing && r.plan === nothing
                    @test r.act_now_action === nothing && r.evpi === nothing
                    @test r.value_of_learning === nothing
                    @test_throws ArgumentError run(; method=:points, passive=true)
                else
                    @test r.no_learning == exact.no_learning && r.plan == exact.plan
                    @test r.act_now_action == exact.act_now_action
                    @test r.no_learning <= r.active
                    @test run(; method=:points, grid=3, passive=true).passive ==
                          exact.passive
                end
            end
            # The exact methods report no gap.
            @test exact.active_upper == exact.active
            # One period needs no grid: the first action is chosen on the prior itself.
            horizon == 1 && @test coarse.active == exact.active == coarse.active_upper
            # A finer grid is usually tighter (counted below; it is not a theorem).
            tight += fine.active_upper == fine.active
            narrower += fine.active_upper - fine.active <
                        coarse.active_upper - coarse.active
            # Floating point: the same bounds to rounding.
            fl = adaptive_management(DecisionProblem(PayoffTable(base.payoff.actions,
                                                                 base.payoff.states,
                                                                 Float64.(base.payoff.Q)),
                                                     Float64.(probabilities(base.belief))),
                                     [a => Experiment(Float64.(e.likelihood))
                                      for (a, e) in given]; horizon,
                                     discount=Float64(discount),
                                     transitions=transitions === nothing ? nothing :
                                                 transitions isa AbstractMatrix ?
                                                 Float64.(transitions) :
                                                 [a => Float64.(T)
                                                  for (a, T) in transitions], signal_of,
                                     method=:points, grid=3)
            @test fl.active ≈ coarse.active atol = 1e-8
            @test fl.active_upper ≈ coarse.active_upper atol = 1e-8
        end
        @test tight > 0 && narrower > 10
    end

    @testset "two states: against the exact vectors over a long horizon" begin
        table = PayoffTable([:none, :inspect, :treat], [:clear, :infested],
                            BigRat[50 20; 48 18; 40 10])
        untreated = BigRat[23//25 0; 2//25 1]
        treated = BigRat[23//25 391//500; 2//25 109//500]
        moves = (; none=untreated, inspect=untreated, treat=treated)
        noticed(d) = Experiment(BigRat[1 1-d; 0 d]; signals=[:not_seen, :seen])
        seen = (; none=noticed(1 // 5), inspect=noticed(9 // 10), treat=noticed(9 // 10))
        problem = DecisionProblem(table, BigRat[9 // 10, 1 // 10]; baseline=:none)
        exact = adaptive_management(problem, seen; horizon=30, transitions=moves, depth=0,
                                    method=:vectors, passive=false)
        approx = adaptive_management(problem, seen; horizon=30, transitions=moves, depth=0,
                                     method=:points, grid=20)
        @test approx.active <= exact.active <= approx.active_upper
        @test approx.active_upper - approx.active < exact.active / 200
        @test approx.first_action == exact.first_action
    end

    @testset "what is refused" begin
        two = DecisionProblem(PayoffTable([:a, :b], [:x, :y], [1 0; 0 1]), [1 // 2, 1 // 2])
        shows = Experiment([1 0; 0 1])
        @test_throws ArgumentError adaptive_management(two, shows; horizon=2,
                                                       method=:points, grid=0)
        # Integer payoffs and an integer prior are carried as rationals.
        sure = DecisionProblem(PayoffTable([:a, :b], [:x, :y], [1 0; 0 1]), [1, 0])
        r = adaptive_management(sure, shows; horizon=3, method=:points, grid=2)
        @test r.active == r.active_upper == 3 && r.active isa Rational
    end
end
