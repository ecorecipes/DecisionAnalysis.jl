# Count tables as payoffs, gains under imprecise beliefs, the Pareto front of named
# tables, the "can this information change the decision" check, and the tabular
# displays.

# Oracle: is there one action that is optimal after every signal of positive
# probability? Written with the posterior, which the package never forms.
function oracle_one_action_always_best(Q, p, L)
    for a in axes(Q, 1)
        ok = true
        for z in axes(L, 1)
            pz = sum(p[t] * L[z, t] for t in axes(L, 2))
            iszero(pz) && continue
            posterior = [p[t] * L[z, t] / pz for t in axes(L, 2)]
            eu = oracle_expected_utility(Q, posterior)
            all(b -> eu[b] <= eu[a], eachindex(eu)) || (ok = false)
        end
        ok && return true
    end
    return false
end

# Oracle: the Pareto front of a score matrix, by pairwise comparison.
function oracle_pareto(S)
    keep = Int[]
    for a in axes(S, 1)
        beaten = false
        for b in axes(S, 1)
            ge = all(S[b, j] >= S[a, j] for j in axes(S, 2))
            gt = any(S[b, j] > S[a, j] for j in axes(S, 2))
            beaten |= ge && gt
        end
        beaten || push!(keep, a)
    end
    return keep
end

plain(x) = sprint(show, MIME"text/plain"(), x)
markdown(x) = sprint(show, MIME"text/markdown"(), x)
html(x) = sprint(show, MIME"text/html"(), x)
widest(text) = maximum(length, split(text, '\n'))

@testset "count tables, gain ranges, displays" begin
    actions = [:no_intervention, :salt, :baths, :disinfect]
    states = [:absent, :low, :high]
    uninfected = PayoffTable(actions, states,
                             [500 300 100; 490 320 160; 470 380 250; 350 340 320])
    infected = PayoffTable(actions, states, [0 150 250; 0 120 200; 0 60 120; 0 10 20])
    prior = FiniteBelief(states, [1 // 2, 3 // 10, 1 // 5])

    @testset "arithmetic on payoff tables" begin
        @test (uninfected + infected).Q == uninfected.Q .+ infected.Q
        @test (uninfected - infected).Q == uninfected.Q .- infected.Q
        @test (-infected).Q == -infected.Q
        @test (2 * uninfected).Q == 2 .* uninfected.Q
        @test (uninfected * (1 // 2)).Q == uninfected.Q .// 2
        @test (uninfected / 2).Q == uninfected.Q ./ 2
        @test uninfected + infected isa PayoffTable{Int}
        @test (uninfected + infected).actions == actions
        @test (uninfected - infected).states == states
        @test uninfected - uninfected == 0 * uninfected
        other = PayoffTable(reverse(actions), states, infected.Q)
        @test_throws LabelMismatchError uninfected + other
        @test_throws LabelMismatchError uninfected -
                                        PayoffTable(actions, [:a, :b, :c], infected.Q)
        @test_throws ArgumentError uninfected / 0
        # Standard errors: scaled by a multiple, dropped by a sum.
        se = PayoffTable(actions, states, Float64.(uninfected.Q);
                         standard_errors=fill(2.0, 4, 3))
        @test (-3 * se).standard_errors == fill(6.0, 4, 3)
        @test (se / 2).standard_errors == fill(1.0, 4, 3)
        @test (-se).standard_errors == fill(2.0, 4, 3)
        @test (se + se).standard_errors === nothing
        rng = Random.Xoshiro(20261201)
        for _ in 1:30
            m, n = rand(rng, 1:4), rand(rng, 1:4)
            a = random_payoff_table(rng, m, n; T=BigRat)
            b = random_payoff_table(rng, m, n; T=BigRat)
            w, v = rand(rng, -5:5) // rand(rng, 1:4), rand(rng, -5:5)
            got = combine_payoffs(a => w, b => v)
            @test got isa PayoffTable{BigRat}
            for i in 1:m, j in 1:n
                @test got.Q[i, j] == w * a.Q[i, j] + v * b.Q[i, j]
            end
            @test combine_payoffs(a => 1) == a
            @test combine_payoffs(a => 1, b => -1) == a - b
        end
        @test_throws ArgumentError combine_payoffs()
        @test_throws LabelMismatchError combine_payoffs(uninfected => 1, other => 1)
    end

    @testset "individuals_payoff and infection_payoff" begin
        @test individuals_payoff(uninfected) == uninfected
        outcomes = (; uninfected, infected)
        @test individuals_payoff(outcomes, :uninfected) == uninfected
        @test individuals_payoff(Dict(:n => uninfected), :n) == uninfected
        @test individuals_payoff([:n => uninfected, :i => infected], "i") == infected
        @test_throws ArgumentError individuals_payoff(outcomes, :missing)
        @test_throws ArgumentError individuals_payoff((; n=1), :n)
        @test_throws InvalidPayoffError individuals_payoff(-infected)
        for w in (0, 1 // 2, 1, 3)
            got = infection_payoff(; uninfected, infected, weight=w)
            @test got.Q == uninfected.Q .- w .* infected.Q
            @test got == infection_payoff(outcomes; weight=w)
            @test got == combine_payoffs(uninfected => 1, infected => -w)
        end
        @test infection_payoff(; uninfected, infected) == uninfected - infected
        @test infection_payoff(; uninfected, infected, weight=0) == uninfected
        @test infection_payoff((; healthy=uninfected, sick=infected); uninfected=:healthy,
                               infected=:sick) == uninfected - infected
        @test infection_payoff(; uninfected, infected, weight=1 // 3) isa
              PayoffTable{Rational{Int}}
        @test_throws ArgumentError infection_payoff(; uninfected, infected, weight=-1)
        @test_throws InvalidPayoffError infection_payoff(; uninfected=-uninfected, infected)
        @test_throws InvalidPayoffError infection_payoff(; uninfected, infected=-infected)
        @test_throws LabelMismatchError infection_payoff(; uninfected,
                                                         infected=PayoffTable(infected.Q))
        # Linear utility: the expected utility is the expected count.
        p = DecisionProblem(infection_payoff(; uninfected, infected, weight=2), prior)
        eu = aggregate(Expectation(), p)
        u = aggregate(Expectation(), DecisionProblem(uninfected, prior))
        i = aggregate(Expectation(), DecisionProblem(infected, prior))
        @test eu == u .- 2 .* i
    end

    @testset "Pareto front of named tables sharing one belief" begin
        objectives = (; uninfected, fewer_infected=-infected)
        front = decide(objectives, prior, ParetoFront())
        by_hand = decide([DecisionProblem(uninfected, prior),
                          DecisionProblem(-infected, prior)], ParetoFront())
        @test front.chosen == by_hand.chosen
        @test front.scores == by_hand.scores
        @test front.details.objectives == [:uninfected, :fewer_infected]
        @test by_hand.details.objectives == [:objective1, :objective2]
        @test front.chosen == actions[oracle_pareto(front.scores)]
        # Other containers, a plain probability vector, and the keywords.
        pairs_form = decide([:u => uninfected, :i => -infected], [1 // 2, 3 // 10, 1 // 5],
                            ParetoFront())
        @test pairs_form.chosen == front.chosen
        @test pairs_form.details.objectives == [:u, :i]
        dict_form = decide(Dict("u" => uninfected), prior, ParetoFront())
        @test dict_form.chosen ==
              decide(DecisionProblem(uninfected, prior),
                     MaxExpectedUtility()).chosen
        some = decide(objectives, prior, ParetoFront(); feasible=[:salt, :baths])
        @test some.actions == [:salt, :baths]
        worst = decide(objectives, ScenarioSet(states), ParetoFront(WorstCase()))
        @test worst.chosen == [:disinfect]
        @test_throws ArgumentError decide((;), prior, ParetoFront())
        @test_throws ArgumentError decide((; a=1), prior, ParetoFront())
        @test_throws LabelMismatchError decide((; uninfected,
                                                other=PayoffTable(reverse(actions), states,
                                                                  infected.Q)), prior,
                                               ParetoFront())
        @test_throws ArgumentError decide([DecisionProblem(uninfected, prior)],
                                          ParetoFront(); objectives=[:a, :b])
        rng = Random.Xoshiro(20261202)
        for _ in 1:40
            m, n, k = rand(rng, 2:6), rand(rng, 2:4), rand(rng, 1:3)
            belief = random_belief(rng, n; T=BigRat)
            tables = [Symbol(:o, j) => random_payoff_table(rng, m, n; T=BigRat)
                      for j in 1:k]
            r = decide(tables, belief, ParetoFront())
            S = [sum(belief.probabilities[t] * tables[j][2].Q[a, t] for t in 1:n)
                 for a in 1:m, j in 1:k]
            @test r.scores == S
            @test r.chosen == r.actions[oracle_pareto(S)]
        end
    end

    @testset "gain over the baseline under imprecise beliefs" begin
        rng = Random.Xoshiro(20261203)
        for _ in 1:60
            m, n, K = rand(rng, 2:5), rand(rng, 2:4), rand(rng, 1:4)
            table = random_payoff_table(rng, m, n; T=BigRat)
            base = rand(rng, 1:m)
            credal = random_credal_set(rng, n, K; T=BigRat)
            problem = DecisionProblem(table, credal; baseline=table.actions[base])
            range = expected_gain_over_baseline(problem)
            @test range isa NamedTuple{(:lower, :upper)}
            members = [oracle_expected_utility(table.Q, q) for q in extreme_points(credal)]
            for a in 1:m
                gains = [e[a] - e[base] for e in members]
                @test range.lower[a] == minimum(gains)
                @test range.upper[a] == maximum(gains)
                # Linear in the prior: no mixture leaves the range.
                for w in simplex_grid(K, 4)
                    g = sum(w[k] * gains[k] for k in 1:K)
                    @test range.lower[a] <= g <= range.upper[a]
                end
            end
            @test iszero(range.lower[base]) && iszero(range.upper[base])
            # A one-point credal set gives the single-prior expected gain twice.
            q = first(extreme_points(credal))
            one = expected_gain_over_baseline(DecisionProblem(table, CredalSet([q]);
                                                              baseline=table.actions[base]))
            single = expected_gain_over_baseline(DecisionProblem(table, q;
                                                                 baseline=table.actions[base]))
            @test one.lower == single && one.upper == single
            # Scenario set: the minimum and maximum gain over the states kept.
            kept = sort(unique(rand(rng, 1:n, rand(rng, 1:n))))
            scen = DecisionProblem(table, ScenarioSet(table.states[kept]);
                                   baseline=table.actions[base])
            sr = expected_gain_over_baseline(scen)
            for a in 1:m
                gains = [table.Q[a, t] - table.Q[base, t] for t in kept]
                @test sr.lower[a] == minimum(gains) && sr.upper[a] == maximum(gains)
            end
        end
        t = PayoffTable([:wait, :act], [:low, :high], [0 -100; -30 -30])
        @test_throws MissingBaselineError expected_gain_over_baseline(DecisionProblem(t,
                                                                                      ScenarioSet([:low])))
        # A filtered problem lists the feasible actions only.
        barred = DecisionProblem(t, CredalSet([:low, :high], [[1 // 2, 1 // 2]]);
                                 baseline=:wait, feasible=[:act])
        @test expected_gain_over_baseline(barred) == (lower=[20], upper=[20])
    end

    @testset "information_can_change_decision ⟺ EVSI > 0" begin
        p = DecisionProblem(PayoffTable([0 -100; -30 -30]), [3 // 5, 2 // 5])
        @test information_can_change_decision(p, Experiment([4//5 1//5; 1//5 4//5]))
        @test !information_can_change_decision(p, Experiment([1//2 1//2; 1//2 1//2]))
        # A test too weak to move the choice is worth nothing, although it is
        # informative.
        weak = Experiment([11//20 9//20; 9//20 11//20])
        @test !information_can_change_decision(p, weak)
        @test evsi(p, weak) == 0
        rng = Random.Xoshiro(20261204)
        changes = 0
        for trial in 1:400
            m, n, nz = rand(rng, 1:4), rand(rng, 2:4), rand(rng, 1:4)
            # Small integer payoffs and coarse likelihoods, so that ties, dominant
            # actions and uninformative signals all occur; some states and signals
            # have probability zero.
            Q = BigRat.(rand(rng, 0:3, m, n))
            w = rand(rng, 0:3, n)
            all(iszero, w) && (w[1] = 1)
            pr = BigRat.(w) .// sum(w)
            L = trial % 5 == 0 ? repeat(random_stochastic(rng, BigRat, nz, 1), 1, n) :
                random_stochastic(rng, BigRat, nz, n)
            problem = DecisionProblem(PayoffTable(Q), pr)
            e = Experiment(L)
            can = information_can_change_decision(problem, e)
            @test can isa Bool
            @test can == (evsi(problem, e) > 0)
            @test can == (oracle_evsi(Q, pr, L) > 0)
            @test can == !oracle_one_action_always_best(Q, pr, L)
            changes += can
        end
        @test 40 < changes < 360
        # The identity experiment changes the decision exactly when EVPI is positive.
        for _ in 1:50
            problem = exact_problem(rng, rand(rng, 1:4), rand(rng, 2:4))
            n = size(problem.payoff, 2)
            identity_experiment = Experiment(BigRat[i == j for i in 1:n, j in 1:n])
            @test information_can_change_decision(problem, identity_experiment) ==
                  (evpi(problem) > 0)
        end
        # Feasibility, a known state, floating point, and the belief forms it refuses.
        t = PayoffTable([:wait, :act], [:low, :high], [0 -100; -30 -30])
        e = Experiment([4//5 1//5; 1//5 4//5])
        @test !information_can_change_decision(DecisionProblem(t, [3 // 5, 2 // 5];
                                                               feasible=[:act]), e)
        @test !information_can_change_decision(DecisionProblem(t, PointBelief(:high)), e)
        fl = DecisionProblem(PayoffTable([0.0 -100.0; -30.0 -30.0]), [0.6, 0.4])
        @test information_can_change_decision(fl, Experiment([0.8 0.2; 0.2 0.8]))
        @test !information_can_change_decision(fl, Experiment([0.5 0.5; 0.5 0.5]))
        @test_throws BeliefFormError information_can_change_decision(DecisionProblem(t,
                                                                                     ScenarioSet([:low])),
                                                                     e)
        @test_throws InvalidExperimentError information_can_change_decision(DecisionProblem(t,
                                                                                            [3 //
                                                                                             5,
                                                                                             2 //
                                                                                             5]),
                                                                            Experiment([1//1 0 0;
                                                                                        0 1 1]))
    end

    @testset "minimax regret carries its menu-dependence note" begin
        problem = DecisionProblem(uninfected, prior)
        r = decide(problem, MinimaxRegret())
        @test r.details.menu_dependent
        @test occursin("menu of actions", r.details.note)
        @test occursin("menu of actions", plain(r))
        relative = decide(problem,
                          DecisionRule(WorstCase();
                                       transform=RelativeRegretTransform()))
        @test relative.details.menu_dependent
        @test decide(problem, Maximin()).details == (;)
        @test decide(problem, MaxExpectedUtility()).details == (;)
        @test !occursin("menu", plain(decide(problem, Maximin())))
    end

    @testset "regret tables carry their labels" begin
        problem = DecisionProblem(uninfected, prior; feasible=[:salt, :baths, :disinfect])
        R = regret_table(problem)
        @test R isa RegretTable{Int} && R isa AbstractMatrix{Int}
        @test action_labels(R) == [:salt, :baths, :disinfect]
        @test state_labels(R) == states
        @test R == oracle_regret(uninfected.Q[2:4, :])
        @test R[2, 3] == 70 && size(R) == (3, 3) && !R.relative
        @test R * prior.probabilities == expected_regret(problem)
        @test relative_regret_table(uninfected).relative
        scen = DecisionProblem(uninfected, ScenarioSet([:low, :high]))
        @test state_labels(regret_table(scen)) == [:low, :high]
        @test state_labels(regret_table(DecisionProblem(uninfected, PointBelief(:low)))) ==
              [:low]
        @test regret_table(uninfected) isa RegretTable
    end

    @testset "tabular displays" begin
        problem = DecisionProblem(infection_payoff(; uninfected, infected), prior;
                                  baseline=:no_intervention)
        credal = CredalSet(states, [[1 // 2, 3 // 10, 1 // 5], [1 // 5, 2 // 5, 2 // 5]])
        shown = Any[uninfected, regret_table(problem), relative_regret_table(uninfected),
                    decide(problem, MaxExpectedUtility()), decide(problem, MinimaxRegret()),
                    decide(problem, Satisfice(1000)),
                    decide(DecisionProblem(uninfected, credal), EAdmissible()),
                    decide(DecisionProblem(uninfected, credal), IntervalDominance()),
                    decide((; uninfected, fewer_infected=-infected), prior, ParetoFront()),
                    DecisionReport(problem; credal, hurwicz=1 // 2)]
        for x in shown
            text, md, page = plain(x), markdown(x), html(x)
            @test widest(text) <= 80
            @test !endswith(text, "\n")
            # Every action appears in each of the three forms.
            for a in (x isa DecisionResult ? x.actions : actions)
                @test occursin(String(a), text)
                @test occursin("`$a`", md)
                @test occursin("<code>$a</code>", page)
            end
            # Markdown: a pipe table whose rows all have the same number of cells.
            rows = filter(l -> startswith(l, "|"), split(md, '\n'))
            @test length(rows) >= 3
            @test any(l -> occursin(r"^\|(:--|--:)(\|(:--|--:))*\|$", l), rows)
            @test occursin("<table>", page) && occursin("</table>", page)
            @test count("<tr>", page) ==
                  length(rows) - count(l -> occursin(r"^\|[:\-|]+\|$", l), rows)
            for mime in ("text/plain", "text/markdown", "text/html")
                @test showable(mime, x)
            end
        end
        # The payoff table: a header of states and one aligned row per action.
        lines = split(plain(uninfected), '\n')
        @test lines[1] == "PayoffTable{Int64}: 4 actions × 3 states"
        @test split(lines[2]) == ["action", "absent", "low", "high"]
        @test split(lines[4]) == ["no_intervention", "500", "300", "100"]
        @test length(lines) == 7
        @test occursin("| `no_intervention` | 500 | 300 | 100 |", markdown(uninfected))
        @test occursin("| action | `absent` | `low` | `high` |", markdown(uninfected))
        # Standard errors are shown when there are any.
        se = PayoffTable(actions, states, Float64.(uninfected.Q);
                         standard_errors=fill(2.5, 4, 3))
        @test occursin("Standard errors", plain(se)) && occursin("2.5", plain(se))
        # A wide table is broken into blocks of columns, each with the action labels.
        wide = PayoffTable(collect(1:60)' .* [1, 2] .* 1000;
                           actions=[:first_action, :second_action])
        text = plain(wide)
        @test widest(text) <= 80
        @test count("first_action", text) > 3
        @test occursin("s60", text) && occursin("120000", text)
        # A rule's result: scores, the chosen marks and the first choice.
        text = plain(decide(problem, MaxExpectedUtility()))
        @test occursin("DecisionResult: MaxExpectedUtility (higher is better)", text)
        @test occursin("First choice: baths", text)
        @test occursin("lower is better", plain(decide(problem, MinimaxRegret())))
        @test occursin("no action chosen", plain(decide(problem, Satisfice(1000))))
        tied = decide(DecisionProblem(PayoffTable([:a, :b], [:s], reshape([1, 1], 2, 1)),
                                      [1]), MaxExpectedUtility())
        @test occursin("First choice: a (tied: a, b)", plain(tied))
        @test occursin("lower expectation",
                       plain(decide(DecisionProblem(uninfected, credal),
                                    IntervalDominance())))
        @test occursin("on the front",
                       plain(decide((; uninfected, fewer_infected=-infected), prior,
                                    ParetoFront())))
        # HTML is escaped.
        odd = PayoffTable([Symbol("a<b")], [Symbol("x&y")], reshape([1], 1, 1))
        @test occursin("a&lt;b", html(odd)) && occursin("x&amp;y", html(odd))
        @test !occursin("a<b", html(odd))
        # The compact one-line forms are unchanged.
        @test sprint(show, uninfected) == "PayoffTable{Int64}(4 actions × 3 states)"
        @test occursin("DecisionProgram{Rational{BigInt}}(e_admissibility: 3 variables, 5 constraints)",
                       sprint(show,
                              eadmissibility_model(DecisionProblem(uninfected, credal),
                                                   :salt)))
        @test occursin("exact",
                       sprint(show,
                              garbling_result(Experiment([1//1 0; 0 1]),
                                              Experiment([1//1 0; 0 1]))))
        @test occursin("optimal",
                       sprint(show,
                              solve_program(eadmissibility_model(DecisionProblem(uninfected,
                                                                                 credal),
                                                                 :salt))))
    end

    @testset "report: experiments, partitions, objectives, gain ranges" begin
        problem = DecisionProblem(infection_payoff(; uninfected, infected), prior;
                                  baseline=:no_intervention)
        swab = Experiment([9//10 3//10 1//10; 1//10 7//10 9//10]; signals=[:neg, :pos],
                          states, cost=5)
        coin = Experiment([1//2 1//2 1//2; 1//2 1//2 1//2]; states)
        partitions = (; presence=[:absent, :present, :present],
                      severity=[:mild, :mild, :severe])
        report = DecisionReport(problem; experiments=(; swab, coin), partitions,
                                objectives=(; uninfected, fewer_infected=-infected))
        @test [e.name for e in report.experiments] == [:swab, :coin]
        @test report.experiments[1].evsi == evsi(problem, swab)
        @test report.experiments[1].cost == 5
        @test report.experiments[1].net == enbs(problem, swab)
        @test report.experiments[1].can_change_decision ==
              information_can_change_decision(problem, swab)
        @test report.experiments[2].evsi == 0
        @test !report.experiments[2].can_change_decision
        @test report.evsi == evsi(problem, swab)
        @test [q.name for q in report.partitions] == [:presence, :severity]
        @test [q.evppi for q in report.partitions] ==
              [evppi(problem, partitions.presence), evppi(problem, partitions.severity)]
        @test report.pareto.chosen ==
              decide((; uninfected, fewer_infected=-infected), prior, ParetoFront()).chosen
        @test report.pareto.details.objectives == [:uninfected, :fewer_infected]
        text = plain(report)
        @test widest(text) <= 80
        for needle in ("EVSI of swab", "EVSI of coin", "EVPPI of presence",
                       "EVPPI of severity", "this experiment cannot change the decision",
                       "Experiments net of their cost", "Pareto front", "Not dominated",
                       "Minimax regret depends on the menu of actions")
            @test occursin(needle, text)
        end
        @test occursin("this experiment cannot change the decision", markdown(report))
        @test occursin("<table>", html(report))
        @test any(a -> occursin("menu of actions", a), report.assumptions)
        @test any(a -> occursin("Pareto front", a), report.assumptions)
        @test any(a -> occursin("EVPPI", a), report.assumptions)
        # The single `experiment` keyword still works, beside the named ones.
        both = DecisionReport(problem; experiment=swab, experiments=[:coin => coin])
        @test [e.name for e in both.experiments] == [:experiment, :coin]
        @test both.evsi == evsi(problem, swab)
        @test_throws ArgumentError DecisionReport(problem; experiment=swab,
                                                  experiments=(; experiment=coin))
        @test_throws ArgumentError DecisionReport(problem; experiments=(; swab=1))
        @test_throws ArgumentError DecisionReport(problem; pareto=ParetoFront())
        @test_throws ArgumentError DecisionReport(problem; partitions=(; bad=[1, 2]))
        rows = collect(Tables.rows(report))
        @test [r.ParetoFront for r in rows] == [a in report.pareto.chosen for a in actions]

        # A credal set as the belief: the range of the expected gain, and the
        # objectives scored by their lower expectation.
        credal = CredalSet(states, [[1 // 2, 3 // 10, 1 // 5], [1 // 5, 2 // 5, 2 // 5]])
        cp = DecisionProblem(problem.payoff, credal; baseline=:no_intervention)
        cr = DecisionReport(cp; objectives=(; uninfected, fewer_infected=-infected))
        @test cr.expected_gain === nothing
        @test cr.gain_range == expected_gain_over_baseline(cp)
        @test cr.pareto.rule.aggregator isa LowerExpectation
        text = plain(cr)
        @test occursin("Range of the expected gain over the baseline", text)
        @test occursin("lowest gain", text) && occursin("highest gain", text)
        @test widest(text) <= 80
        rows = collect(Tables.rows(cr))
        @test [r.gain_over_baseline_lower for r in rows] == cr.gain_range.lower
        @test [r.gain_over_baseline_upper for r in rows] == cr.gain_range.upper
        @test any(a -> occursin("range of the expected gain over the credal set", a),
                  cr.assumptions)
        @test_throws BeliefFormError DecisionReport(cp; partitions)
        @test_throws BeliefFormError DecisionReport(cp; experiments=(; swab))
        # A scenario set: the range of the gain over the scenarios, worst-case front.
        sp = DecisionProblem(problem.payoff, ScenarioSet(states); baseline=:no_intervention)
        sr = DecisionReport(sp; objectives=(; uninfected, fewer_infected=-infected))
        @test sr.gain_range == expected_gain_over_baseline(sp)
        @test sr.pareto.rule.aggregator isa WorstCase
        @test occursin("Range of the gain over the baseline", plain(sr))
        @test DecisionReport(sp; objectives=(; uninfected),
                             pareto=ParetoFront(BestCase())).pareto.chosen ==
              [:no_intervention]
        # The robust sets by another route are the same sets.
        byhighs = DecisionReport(problem; credal, optimizer=HIGHS)
        byinternal = DecisionReport(problem; credal)
        @test [r.chosen for r in byhighs.robust] == [r.chosen for r in byinternal.robust]
        @test byhighs.robust_evpi.upper == byinternal.robust_evpi.upper
    end
end
