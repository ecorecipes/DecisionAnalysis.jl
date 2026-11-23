@testset "decision report" begin
    examples = load_examples()
    fx = first(examples)
    flood = fixture_problem(fx, identity)
    credal = CredalSet(flood.payoff.states, [rational_vector(q) for q in fx["credal"]])
    gauge = fixture_experiment(fx["experiments"]["gauge"], identity)
    report = DecisionReport(flood; credal, experiment=gauge,
                            assumptions=["Costs are in millions of pounds."])
    names(r) = [DecisionAnalysis._rule_name(res.rule) for res in r]

    @testset "flood: all five rules side by side" begin
        @test issubset([:MaxExpectedUtility, :Maximin, :MinimaxRegret, :GammaMaximin],
                       names(report.results))
        @test names(report.robust) == [:EAdmissible, :Maximal, :IntervalDominance]
        choice = Dict(DecisionAnalysis._rule_name(res.rule) => first_choice(res)
                      for res in report.results)
        @test choice[:MaxExpectedUtility] == :do_nothing
        @test choice[:Maximin] == :relocate_assets
        @test choice[:MinimaxRegret] == :raise_levee
        @test choice[:GammaMaximin] == :raise_levee
        @test report.robust[1].chosen == [:do_nothing, :raise_levee]
        @test report.max_regret == [102, 42, 48]
        @test report.expected_regret == expected_regret(flood)
        @test report.evpi == 207 // 20
        @test report.evsi == 97 // 20
        @test report.robust_evpi.lower <= report.evpi <= report.robust_evpi.upper
        @test report.expected_gain == expected_gain_over_baseline(flood)
        @test "Costs are in millions of pounds." in report.assumptions
    end

    @testset "show" begin
        text = sprint(show, MIME"text/plain"(), report)
        for needle in ("MaxExpectedUtility", "Maximin", "MinimaxRegret", "GammaMaximin",
                       "EAdmissible", "Regret profile", "EVPI", "EVSI", "Assumptions",
                       "do_nothing", "raise_levee", "relocate_assets", "10.35", "4.85",
                       "Baseline (no intervention): do_nothing",
                       "Expected gain over the baseline", "[the baseline]",
                       "[not the baseline]", "Costs are in millions of pounds.")
            @test occursin(needle, text)
        end
        # Each rule's line says whether its choice is the baseline.
        line(rule) = only(filter(l -> startswith(strip(l), rule) && occursin("baseline]", l),
                                 split(text, '\n')))
        @test occursin("[the baseline]", line("MaxExpectedUtility"))
        @test occursin("[not the baseline]", line("Maximin"))
        @test occursin("[not the baseline]", line("MinimaxRegret"))
        @test occursin("[not the baseline]", line("GammaMaximin"))
        @test occursin("DecisionReport(", sprint(show, report))
        # Without a baseline the header and the assumptions say so.
        plain = DecisionReport(DecisionProblem(flood.payoff, flood.belief))
        text = sprint(show, MIME"text/plain"(), plain)
        @test occursin("Baseline (no intervention): none set", text)
        @test !occursin("Expected gain over the baseline", text)
        @test !occursin("baseline]", text)
        @test any(a -> occursin("No baseline", a), plain.assumptions)
        @test plain.expected_gain === nothing
        @test any(a -> occursin("baseline (no intervention) is do_nothing", a),
                  report.assumptions)
    end

    @testset "Tables.jl" begin
        @test Tables.istable(report)
        rows = collect(Tables.rows(report))
        @test length(rows) == 3
        @test [r.action for r in rows] == [:do_nothing, :raise_levee, :relocate_assets]
        @test [r.MaxExpectedUtility for r in rows] == [-39 // 2, -97 // 4, -48]
        @test [r.MaxExpectedUtility_chosen for r in rows] == [true, false, false]
        @test [r.MinimaxRegret_chosen for r in rows] == [false, true, false]
        @test [r.max_regret for r in rows] == [102, 42, 48]
        @test [r.expected_regret for r in rows] == expected_regret(flood)
        @test [r.expected_gain_over_baseline for r in rows] == [0, -19 // 4, -57 // 2]
        @test [r.EAdmissible for r in rows] == [true, true, false]
        cols = Tables.columntable(report)
        @test cols.action == [:do_nothing, :raise_levee, :relocate_assets]
        @test cols.GammaMaximin == [-33, -113 // 4, -48]
        @test :expected_gain_over_baseline ∉
              propertynames(first(Tables.rows(DecisionReport(DecisionProblem(flood.payoff,
                                                                             flood.belief)))))
    end

    @testset "by belief form" begin
        scen = DecisionReport(DecisionProblem(flood.payoff,
                                              ScenarioSet(flood.payoff.states);
                                              baseline=:do_nothing); hurwicz=1 // 4)
        @test names(scen.results) ==
              [:Maximin, :Maximax, :Laplace, :Hurwicz, :MinimaxRegret]
        @test scen.evpi === nothing && scen.expected_regret === nothing
        @test isempty(scen.robust) && scen.robust_evpi === nothing
        @test scen.expected_gain === nothing
        text = sprint(show, MIME"text/plain"(), scen)
        @test occursin("not defined without probabilities", text)
        @test occursin("Baseline (no intervention): do_nothing", text)
        @test any(a -> occursin("no expected gain is shown", a), scen.assumptions)

        cred = DecisionReport(DecisionProblem(flood.payoff, credal))
        @test :MaxExpectedUtility ∉ names(cred.results)
        @test :GammaMaximin in names(cred.results)
        @test cred.evpi === nothing && cred.robust_evpi isa RobustVOI
        @test occursin("EVPI envelope", sprint(show, MIME"text/plain"(), cred))

        point = DecisionReport(DecisionProblem(flood.payoff, PointBelief(:high)))
        @test point.evpi == 0
        @test first_choice(first(point.results)) == :raise_levee

        filtered = DecisionReport(DecisionProblem(flood.payoff, flood.belief;
                                                  feasible=[:raise_levee, :relocate_assets],
                                                  baseline=:do_nothing))
        @test length(collect(Tables.rows(filtered))) == 2
        @test any(a -> occursin("Infeasible actions", a), filtered.assumptions)
        @test any(a -> occursin("which is not feasible", a), filtered.assumptions)

        fl = DecisionReport(fixture_problem(fx, Float64))
        @test fl.evpi ≈ 10.35
        @test occursin("10.35", sprint(show, MIME"text/plain"(), fl))

        @test_throws ArgumentError DecisionReport(DecisionProblem(flood.payoff, credal);
                                                  credal)
        @test_throws ArgumentError DecisionReport(DecisionProblem(flood.payoff,
                                                                  ScenarioSet([:low]));
                                                  credal)
        @test_throws BeliefFormError DecisionReport(DecisionProblem(flood.payoff, credal);
                                                    experiment=gauge)
    end

    @testset "show methods of the other types" begin
        @test occursin("FiniteBelief", sprint(show, flood.belief))
        @test occursin("3 points", sprint(show, credal))
        @test occursin("ScenarioSet", sprint(show, ScenarioSet([:a])))
        @test occursin("3 actions", sprint(show, flood.payoff))
        @test occursin("3 feasible of 3", sprint(show, flood))
        @test occursin("2 signals", sprint(show, gauge))
        @test sprint(show, MinimaxRegret()) == "MinimaxRegret"
        @test occursin("raise_levee", sprint(show, decide(flood, MinimaxRegret())))
        @test occursin("no action chosen",
                       sprint(show, decide(flood, Satisfice(0; aggregator=WorstCase()))))
        @test occursin("envelope", sprint(show, report.robust_evpi))
    end
end
