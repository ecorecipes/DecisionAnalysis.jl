# Survey first or act now, and the selection of units under a budget (src/survey.jl),
# after Raymond et al. 2020 (J Appl Ecol 57:819-830, doi 10.1111/1365-2664.13580). The
# paper's own results cannot be recomputed (its 238,247 units are not archived), so the
# tests are its one worked example (supplement, Figure B2), the closed form of its rule
# (Lean: `survey_first_iff`), enumeration over every policy, and synthetic units.

# One species, one unit: protecting is worth 1 where the species is present.
function presence_problem(p)
    table = PayoffTable([:none, :protect], [:absent, :present], [0 0; 0 1])
    return DecisionProblem(table, [1 - p, p]; baseline=:none)
end
# A survey that finds a present species with probability d and never an absent one.
detection(d) = Experiment([1 1-d; 0 d]; signals=[:not_found, :found])
const FOUND_ONLY = [:not_found => :none, :found => :protect]

# Oracle: the value and expected cost of every policy, from the definitions.
function oracle_policies(Q, p, L, costs, base, survey_cost)
    m, nz = size(Q, 1), size(L, 1)
    return map(Iterators.product(ntuple(_ -> 1:m, nz)...)) do policy
        value = sum(p[t] * L[z, t] * (Q[policy[z], t] - Q[base, t])
                    for z in 1:nz, t in axes(Q, 2))
        cost = survey_cost +
               sum(p[t] * L[z, t] * costs[policy[z]] for z in 1:nz, t in axes(Q, 2))
        return (; policy, value, cost)
    end
end

# Oracle: go down the ranking and take every unit that fits.
function oracle_fill(values, costs, budget)
    order = sort(eachindex(values); by=i -> values[i] / costs[i], rev=true)
    taken = Int[]
    for i in order
        if values[i] > 0 && costs[i] <= budget
            budget -= costs[i]
            push!(taken, i)
        end
    end
    return taken
end

plain_unit(value, cost) = (; choice=:act_now, value, cost, largest_cost=cost)

@testset "survey first or act now" begin
    @testset "the worked example of Raymond et al. 2020" begin
        # Figure B2 of the supplement: prior 0.10, detection 0.70; the probability of
        # presence is 1.0 after a detection and 0.03 after none. No cost is printed.
        p, d = 1 // 10, 7 // 10
        not_found = (1 - d) * p / ((1 - d) * p + (1 - p))
        @test not_found == 1 // 31
        @test round(Float64(not_found); digits=2) == 0.03
        u = survey_or_act(presence_problem(p), detection(d); action_costs=[0, 5000],
                          survey_cost=750, policy=FOUND_ONLY)
        @test u.act_now == (; action=:protect, value=1 // 10, cost=5000,
                            cost_effectiveness=1 // 50000)
        @test u.survey.value == 7 // 100
        @test u.survey.probability_of_acting == 7 // 100
        @test u.survey.cost == 7 // 100 * 5000 + 750
        @test u.survey.largest_cost == 5750
        @test u.survey.cost_effectiveness == (7 // 100) / 1100
        @test u.survey_first && u.choice == :survey
        @test (u.value, u.cost, u.largest_cost) == (7 // 100, 1100, 5750)
        # The policy of the paper is the one found when none is imposed.
        free = survey_or_act(presence_problem(p), detection(d); action_costs=[0, 5000],
                             survey_cost=750)
        @test free == u
        @test free.survey.policy == FOUND_ONLY
        # Costs as pairs.
        @test survey_or_act(presence_problem(p), detection(d);
                            action_costs=[:protect => 5000, :none => 0],
                            survey_cost=750) == u
    end

    @testset "the closed form of the rule" begin
        # Survey first exactly when d C (1 − p) > M (Lean: survey_first_iff); on the
        # boundary, act now (survey_first_tie). Costs: the survey cost of the paper
        # and the smallest, mean and largest purchase costs it reports, among others.
        M = 750
        surveyed = 0
        for p in (1 // 20):(1 // 20):(19 // 20), d in (1 // 20):(1 // 20):(19 // 20),
            C in (1000, 2573, 7146, 28560, 100000)

            problem, survey = presence_problem(p), detection(d)
            imposed = survey_or_act(problem, survey; action_costs=[0, C], survey_cost=M,
                                    policy=FOUND_ONLY)
            @test imposed.survey.value == d * p
            @test imposed.survey.cost == d * p * C + M
            @test imposed.act_now.cost_effectiveness == p / C
            @test imposed.survey_first == (d * C * (1 - p) > M)
            free = survey_or_act(problem, survey; action_costs=[0, C], survey_cost=M)
            @test free.survey_first == imposed.survey_first
            @test free.choice == imposed.choice
            (free.value, free.cost) == (imposed.value, imposed.cost) || @test false
            surveyed += free.survey_first
        end
        @test 0 < surveyed < 19 * 19 * 5
        # A tie: 0.7 × C × 0.9 = 750 at C = 25000 / 21.
        tie = survey_or_act(presence_problem(1 // 10), detection(7 // 10);
                            action_costs=[0, 25000 // 21], survey_cost=750)
        @test tie.survey.cost_effectiveness == tie.act_now.cost_effectiveness
        @test tie.choice == :act_now
        # Either side of the boundary at the two values in the caption of the paper's
        # Figure B4 (detection 0.70): prior 0.6842 and cost 19029. The boundaries
        # (cost 3392.7, prior 0.9437) are ours, from the closed form.
        choice(p, C) = survey_or_act(presence_problem(p), detection(7 // 10);
                                     action_costs=[0, C], survey_cost=750).choice
        @test choice(6842 // 10000, 3392) == :act_now
        @test choice(6842 // 10000, 3393) == :survey
        @test choice(9437 // 10000, 19029) == :act_now
        @test choice(9436 // 10000, 19029) == :survey
    end

    @testset "the policy against every policy" begin
        rng = Random.Xoshiro(20261125)
        kept_baseline = 0
        for trial in 1:150
            m, n, nz = rand(rng, 2:4), rand(rng, 2:4), rand(rng, 2:3)
            table = random_payoff_table(rng, m, n; T=BigRat)
            belief = random_belief(rng, n; T=BigRat)
            problem = DecisionProblem(table, belief; baseline=:a1)
            survey = Experiment(random_stochastic(rng, BigRat, nz, n))
            costs = BigRat[0; rand(rng, 1:30, m - 1)]
            M = BigRat(rand(rng, 1:10))
            u = survey_or_act(problem, survey; action_costs=costs, survey_cost=M)
            p = probabilities(belief)
            all_policies = oracle_policies(table.Q, p, survey.likelihood, costs, 1, M)
            @test u.survey.cost_effectiveness ==
                  maximum(x.value / x.cost for x in all_policies)
            mine = Tuple(findfirst(==(a), table.actions) for (_, a) in u.survey.policy)
            same = only(x for x in all_policies if x.policy == mine)
            @test (u.survey.value, u.survey.cost) == (same.value, same.cost)
            kept_baseline += all(==(1), mine)
            gains = [oracle_expected_utility(table.Q, p)[a] -
                     oracle_expected_utility(table.Q, p)[1] for a in 1:m]
            @test u.act_now.cost_effectiveness == maximum(gains[a] / costs[a] for a in 2:m)
            @test u.survey_first ==
                  (u.survey.cost_effectiveness > u.act_now.cost_effectiveness)
            # Floating point reaches the same ratio.
            fl = survey_or_act(DecisionProblem(PayoffTable(table.actions, table.states,
                                                           Float64.(table.Q)),
                                               Float64.(p); baseline=:a1),
                               Experiment(Float64.(survey.likelihood));
                               action_costs=Float64.(costs), survey_cost=Float64(M))
            @test fl.survey.cost_effectiveness ≈ u.survey.cost_effectiveness atol = 1e-12
        end
        @test 0 < kept_baseline < 150
    end

    @testset "three species in one unit" begin
        # The multi-species extension of the paper (supplement, Appendix A): protecting
        # is worth the number of species present, one survey looks for all three with
        # detection 0.6, 0.7 and 0.9 (the paper's), and the unit is bought if at least
        # one is found. Priors are ours. The value here is the expected number of
        # species present in a unit that is bought. The listing in the paper's
        # supplement counts only the species found, Σ d_k p_k, which is smaller: a unit
        # bought because one species was found may hold another that was missed.
        priors = [1 // 5, 1 // 10, 3 // 10]
        detect = [3 // 5, 7 // 10, 9 // 10]
        C, M = 7146, 750
        subsets = collect(Iterators.product(0:1, 0:1, 0:1))[:]
        label(prefix, s) = Symbol(prefix, join(s))
        prior = [prod(s[k] == 1 ? priors[k] : 1 - priors[k] for k in 1:3) for s in subsets]
        Q = vcat(zeros(Int, 1, 8), permutedims([sum(s) for s in subsets]))
        problem = DecisionProblem(PayoffTable([:none, :protect],
                                              [label(:present, s) for s in subsets], Q),
                                  prior; baseline=:none)
        L = [prod(z[k] == 0 ? 1 - s[k] * detect[k] : s[k] * detect[k] for k in 1:3)
             for z in subsets, s in subsets]
        signals = [label(:found, z) for z in subsets]
        survey = Experiment(L; signals)
        any_found = [signals[i] => (sum(subsets[i]) > 0 ? :protect : :none) for i in 1:8]
        u = survey_or_act(problem, survey; action_costs=[0, C], survey_cost=M,
                          policy=any_found)
        q = 1 - prod(1 .- detect .* priors)
        @test u.act_now.value == sum(priors)
        missed(k) = (1 - detect[k]) * prod(1 - detect[j] * priors[j] for j in 1:3 if j != k)
        @test u.survey.value == sum(priors[k] * (1 - missed(k)) for k in 1:3)
        @test sum(detect .* priors) < u.survey.value < u.act_now.value
        @test u.survey.probability_of_acting == q
        @test u.survey.cost == q * C + M
        free = survey_or_act(problem, survey; action_costs=[0, C], survey_cost=M)
        @test free.survey.cost_effectiveness >= u.survey.cost_effectiveness
    end

    @testset "feasibility and signals of probability zero" begin
        table = PayoffTable([:none, :cheap, :dear], [:s1, :s2], [0 0; 1 3; 2 9])
        survey = Experiment([1 0; 0 1; 0 0]; signals=[:low, :high, :never])
        all_three = DecisionProblem(table, [1 // 2, 1 // 2]; baseline=:none)
        u = survey_or_act(all_three, survey; action_costs=[0, 10, 20], survey_cost=1)
        @test u.act_now.action == :dear
        # After :low the cheap action gains 1 for 10, less per unit of cost than the
        # rest of the policy, so the unit is left alone there.
        @test u.survey.policy == [:low => :none, :high => :dear, :never => :none]
        @test (u.survey.value, u.survey.cost) == (9 // 2, 11)
        @test u.survey.largest_cost == 21
        @test u.survey.probability_of_acting == 1 // 2
        # An infeasible action is never used, now or after the survey.
        restricted = DecisionProblem(table, [1 // 2, 1 // 2]; baseline=:none,
                                     feasible=[:none, :cheap])
        r = survey_or_act(restricted, survey; action_costs=[0, 10, 20], survey_cost=1)
        @test r.act_now.action == :cheap
        @test all(a != :dear for (_, a) in r.survey.policy)
        # An action that loses against the baseline is not worth a survey either.
        worse = PayoffTable([:none, :harm], [:s1, :s2], [0 0; -1 -2])
        w = survey_or_act(DecisionProblem(worse, [1 // 2, 1 // 2]; baseline=:none),
                          Experiment([1 0; 0 1]); action_costs=[0, 5], survey_cost=1)
        @test w.act_now.value == -3 // 2
        @test w.survey.value == 0 && w.survey.probability_of_acting == 0
        @test w.choice == :survey && w.value == 0
    end

    @testset "selection under a budget" begin
        # Ranked A, B, C, D, E by value per cost. The first pass takes A and stops at
        # B; the second takes C and stops at D; the third takes E.
        units = [plain_unit(v, c)
                 for (v, c) in
                     [(12, 60), (99 // 10, 55), (9 // 2, 30), (3 // 2, 15), (1 // 2, 10)]]
        full = select_under_budget(units, 100)
        @test full.selected == [1, 3, 5]
        @test (full.value, full.cost, full.remaining) == (17, 100, 0)
        @test full.passes == 3
        two = select_under_budget(units, 100; passes=2)
        @test two.selected == [1, 3]
        @test (two.value, two.cost, two.remaining, two.passes) == (33 // 2, 90, 10, 2)
        @test select_under_budget(units, 100; passes=1).selected == [1]
        @test select_under_budget(units, 0).selected == Int[]
        @test select_under_budget(units, 1000).selected == [1, 2, 3, 4, 5]
        # The order given does not matter when no two units tie.
        @test select_under_budget(reverse(units), 100).selected == [5, 3, 1]
        # Ties: the first pass keeps the order given; later passes take the cheaper.
        tied = [plain_unit(2, 20), plain_unit(1, 10)]
        @test select_under_budget(tied, 20).selected == [1]
        later = [plain_unit(50, 100), plain_unit(24 // 5, 60), plain_unit(2, 40),
                 plain_unit(1, 20)]
        @test select_under_budget(later, 145).selected == [1, 4]
        # A unit that gains nothing is never taken.
        @test select_under_budget([plain_unit(0, 1), plain_unit(-1, 1), plain_unit(1, 5)],
                                  10).selected == [3]

        rng = Random.Xoshiro(20261126)
        for trial in 1:100
            n = rand(rng, 1:12)
            values = BigRat[rand(rng, 1:50) // rand(rng, 1:7) for _ in 1:n]
            costs = BigRat[rand(rng, 1:40) for _ in 1:n]
            allunique(values ./ costs) || continue
            budget = BigRat(rand(rng, 0:150))
            s = select_under_budget(plain_unit.(values, costs), budget)
            @test s.selected == oracle_fill(values, costs, budget)
            @test s.value == sum(values[s.selected]; init=0)
            @test s.cost == sum(costs[s.selected]; init=0) <= budget
            @test s.remaining == budget - s.cost
            @test all(c > s.remaining for (i, c) in enumerate(costs) if !(i in s.selected))
            capped = select_under_budget(plain_unit.(values, costs), budget; passes=2)
            @test issubset(capped.selected, s.selected)
        end
    end

    @testset "units from survey_or_act" begin
        # Synthetic units: priors and costs are ours; the survey cost, the detection
        # probability and the budget are those of the paper, and so is the policy
        # (buy only where the species is found), imposed so that "always survey" means
        # what it does there.
        rng = Random.Xoshiro(20261127)
        priors = [rand(rng, 5:95) // 100 for _ in 1:60]
        costs = rand(rng, 2573:28560, 60)
        units = [survey_or_act(presence_problem(p), detection(7 // 10);
                               action_costs=[0, C], survey_cost=750, policy=FOUND_ONLY)
                 for (p, C) in zip(priors, costs)]
        @test [u.choice == :survey for u in units] ==
              [7 // 10 * C * (1 - p) > 750 for (p, C) in zip(priors, costs)]
        mixed = select_under_budget(units, 100000)
        never = select_under_budget(units, 100000; option=:act_now)
        always = select_under_budget(units, 100000; option=:survey)
        @test mixed.choices == [units[i].choice for i in mixed.selected]
        @test all(==(:act_now), never.choices) && all(==(:survey), always.choices)
        @test never.cost == sum(costs[never.selected]) == never.largest_cost
        @test always.value == sum(7 // 10 * priors[i] for i in always.selected)
        @test always.cost == sum(7 // 10 * priors[i] * costs[i] + 750
                                 for i in always.selected)
        @test always.largest_cost == sum(costs[i] + 750 for i in always.selected)
        @test mixed.cost <= 100000 < mixed.largest_cost
        @test :survey in mixed.choices
        @test mixed.value > never.value
    end

    @testset "what is refused" begin
        problem, survey = presence_problem(1 // 10), detection(7 // 10)
        run(; problem=problem, survey=survey, action_costs=[0, 5000], survey_cost=750,
        policy=nothing) = survey_or_act(problem, survey; action_costs, survey_cost, policy)
        table = PayoffTable([:none, :protect], [:absent, :present], [0 0; 0 1])
        @test_throws MissingBaselineError run(problem=DecisionProblem(table, [0.9, 0.1]))
        @test_throws BeliefFormError run(problem=DecisionProblem(table,
                                                                 CredalSet([:absent,
                                                                            :present],
                                                                           [[0.9, 0.1],
                                                                            [0.5, 0.5]]);
                                                                 baseline=:none))
        @test_throws ArgumentError run(survey_cost=0)
        @test_throws ArgumentError run(action_costs=[10, 5000])
        @test_throws ArgumentError run(action_costs=[0, 0])
        @test_throws ArgumentError run(action_costs=[0, 1, 2])
        @test_throws ArgumentError run(action_costs=[:none => 0])
        @test_throws ArgumentError run(action_costs=[:none => 0, :buy => 1])
        @test_throws ArgumentError run(policy=[:found => :protect])
        @test_throws ArgumentError run(policy=[:found => :protect, :not_found => :sell])
        @test_throws ArgumentError run(problem=DecisionProblem(table, [0.9, 0.1];
                                                               baseline=:none,
                                                               feasible=[:none]))
        @test_throws ArgumentError run(problem=DecisionProblem(table, [0.9, 0.1];
                                                               baseline=:none,
                                                               feasible=[:protect]))
        @test_throws LabelMismatchError run(survey=Experiment([1 0.3; 0 0.7];
                                                              states=[:a, :b]))
        units = [plain_unit(1, 1)]
        @test_throws ArgumentError select_under_budget(units, -1)
        @test_throws ArgumentError select_under_budget(units, 1; passes=0)
        @test_throws ArgumentError select_under_budget(units, 1; option=:best)
    end
end
