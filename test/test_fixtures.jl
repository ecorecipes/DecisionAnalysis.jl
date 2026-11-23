# The three worked examples (lab/wiki/reviews/worked-examples.md), read from the fixture
# file shared with the Lean development. Every quoted figure is reproduced exactly in
# rationals and approximately in Float64.

function fixture_problem(ex, convert_number)
    payoff = PayoffTable(symbols(ex["actions"]), symbols(ex["states"]),
                         convert_number.(rational_matrix(ex["payoff"])))
    return DecisionProblem(payoff, convert_number.(rational_vector(ex["prior"]));
                           baseline=ex["baseline"])
end

function fixture_experiment(spec, convert_number)
    cost = haskey(spec, "cost") ? convert_number(parse_rational(spec["cost"])) : 0
    return Experiment(convert_number.(rational_matrix(spec["likelihood"]));
                      signals=symbols(spec["signals"]), cost)
end

@testset "worked examples" begin
    examples = load_examples()
    @test [ex["name"] for ex in examples] == ["flood", "epidemic", "equipment"]

    @testset "$(ex["name"]), exact" for ex in examples
        problem = fixture_problem(ex, identity)
        want = ex["expected"]
        @test eltype(problem.payoff) == BigRat

        eu = decide(problem, MaxExpectedUtility())
        @test eu.scores == rational_vector(want["expected_utility"])
        @test chosen_indices(eu) == one_based(want["bayes_actions"])
        @test act_now_value(problem) == parse_rational(want["act_now_value"])
        mm = decide(problem, Maximin())
        @test mm.scores == rational_vector(want["maximin"])
        @test chosen_indices(mm) == one_based(want["maximin_actions"])
        @test regret_table(problem) == rational_matrix(want["regret"])
        @test expected_regret(problem) == rational_vector(want["expected_regret"])
        mr = decide(problem, MinimaxRegret())
        @test mr.scores == rational_vector(want["max_regret"]) == max_regret(problem)
        @test chosen_indices(mr) == one_based(want["minimax_regret_actions"])
        @test evpi(problem) == parse_rational(want["evpi"])
        # The identity the reviews quote: EVPI is the expected regret of the Bayes action.
        @test evpi(problem) == expected_regret(problem)[first(chosen_indices(eu))]

        # The baseline is the do-nothing action, the first row of each example.
        @test problem.baseline == Symbol(ex["baseline"]) == first(problem.payoff.actions)
        @test all(iszero, gain_over_baseline(problem).Q[1, :])
        @test expected_gain_over_baseline(problem) == eu.scores .- eu.scores[1]

        experiments = Dict(name => fixture_experiment(spec, identity)
                           for (name, spec) in ex["experiments"])
        for (name, e) in experiments
            @test value_with_information(problem, e) ==
                  parse_rational(want["value_with_information"][name])
            @test evsi(problem, e) == parse_rational(want["evsi"][name])
            @test 0 <= evsi(problem, e) <= evpi(problem)
            spec = ex["experiments"][name]
            if haskey(spec, "cost")
                @test enbs(problem, e) == parse_rational(want["enbs"][name])
            end
            if haskey(spec, "garbling")
                source = experiments[spec["garbling_of"]]
                G = rational_matrix(spec["garbling"])
                @test garble(source, G).likelihood == e.likelihood
                @test is_garbling_of(e, source) == G
                @test is_garbling_of(source, e) === nothing
                @test deficiency(source, e) == 0 < deficiency(e, source)
                @test evsi(problem, e) < evsi(problem, source)
            end
        end

        if haskey(ex, "credal")
            credal = CredalSet(problem.payoff.states,
                               [rational_vector(q) for q in ex["credal"]])
            robust = DecisionProblem(problem.payoff, credal)
            gm = decide(robust, GammaMaximin())
            @test gm.scores == rational_vector(want["lower_expected_utility"])
            @test chosen_indices(gm) == one_based(want["gamma_maximin_actions"])
            @test chosen_indices(decide(robust, EAdmissible())) ==
                  one_based(want["e_admissible"])
        end
    end

    @testset "$(ex["name"]), Float64" for ex in examples
        exact = fixture_problem(ex, identity)
        problem = fixture_problem(ex, Float64)
        want = ex["expected"]
        @test eltype(problem.payoff) == Float64
        approx(x, y) = isapprox(x, Float64.(y); atol=1e-9)
        eu = decide(problem, MaxExpectedUtility())
        @test approx(eu.scores, rational_vector(want["expected_utility"]))
        @test chosen_indices(eu) == one_based(want["bayes_actions"])
        @test approx(regret_table(problem), rational_matrix(want["regret"]))
        @test approx(expected_regret(problem), rational_vector(want["expected_regret"]))
        mr = decide(problem, MinimaxRegret())
        @test approx(mr.scores, rational_vector(want["max_regret"]))
        @test chosen_indices(mr) == one_based(want["minimax_regret_actions"])
        @test chosen_indices(decide(problem, Maximin())) ==
              one_based(want["maximin_actions"])
        @test approx(evpi(problem), parse_rational(want["evpi"]))
        @test approx(expected_gain_over_baseline(problem),
                     expected_gain_over_baseline(exact))
        for (name, spec) in ex["experiments"]
            e = fixture_experiment(spec, Float64)
            @test approx(evsi(problem, e), parse_rational(want["evsi"][name]))
            haskey(spec, "cost") &&
                @test approx(enbs(problem, e), parse_rational(want["enbs"][name]))
            if haskey(spec, "garbling")
                source = fixture_experiment(ex["experiments"][spec["garbling_of"]], Float64)
                @test is_garbling_of(e, source) ≈
                      Float64.(rational_matrix(spec["garbling"])) atol = 1e-6
                @test is_garbling_of(source, e) === nothing
            end
        end
        if haskey(ex, "credal")
            credal = CredalSet(problem.payoff.states,
                               [Float64.(rational_vector(q)) for q in ex["credal"]])
            robust = DecisionProblem(problem.payoff, credal)
            gm = decide(robust, GammaMaximin())
            @test approx(gm.scores, rational_vector(want["lower_expected_utility"]))
            @test chosen_indices(gm) == one_based(want["gamma_maximin_actions"])
            @test chosen_indices(decide(robust, EAdmissible())) ==
                  one_based(want["e_admissible"])
        end
    end

    @testset "the figures as quoted in the reviews" begin
        # Decimal figures copied from the review text, independent of the fixture's
        # "expected" block.
        flood, epidemic, equipment = [fixture_problem(ex, identity) for ex in examples]
        fx, ex2, ex3 = examples
        @test aggregate(Expectation(), flood) == [-195 // 10, -2425 // 100, -48]
        @test aggregate(WorstCase(), flood) == [-150, -90, -48]
        @test max_regret(flood) == [102, 42, 48]
        @test evpi(flood) == 1035 // 100
        gauge = fixture_experiment(fx["experiments"]["gauge"], identity)
        garbled = fixture_experiment(fx["experiments"]["garbled_gauge"], identity)
        @test evsi(flood, gauge) == 485 // 100
        @test evsi(flood, garbled) == 196 // 100
        credal = CredalSet(flood.payoff.states, [rational_vector(q) for q in fx["credal"]])
        robust = DecisionProblem(flood.payoff, credal)
        @test aggregate(LowerExpectation(), robust) == [-33, -2825 // 100, -48]
        @test decide(robust, EAdmissible()).chosen == [:do_nothing, :raise_levee]
        # The five rules of the flood example and what each recommends.
        @test first_choice(decide(flood, MaxExpectedUtility())) == :do_nothing
        @test first_choice(decide(flood, Maximin())) == :relocate_assets
        @test first_choice(decide(flood, MinimaxRegret())) == :raise_levee
        @test first_choice(decide(robust, GammaMaximin())) == :raise_levee

        test2 = fixture_experiment(ex2["experiments"]["test"], identity)
        @test evpi(epidemic) == 18
        @test evsi(epidemic, test2) == 88 // 10
        @test enbs(epidemic, test2) == 58 // 10

        test3 = fixture_experiment(ex3["experiments"]["test"], identity)
        @test evpi(equipment) == 140
        @test evsi(equipment, test3) == 78
        @test enbs(equipment, test3) == 48
        @test expected_regret(equipment) == [240, 140]
        @test max_regret(equipment) == [800, 200]
    end
end
