# Cross-check of the imprecise-probability rules against the Python package `improb`
# 0.1.1 (Matthias Troffaes), run under Python 2.7 by validation/improb_reference.py.
# The test only reads the fixture, so the suite needs no Python. improb holds the
# credal set by the facets of the hull of our generating points and works in exact
# arithmetic, so every comparison is an equality of rationals or of sets. Conventions
# (validation/README.md): dominance is strict on both sides, and every tie is kept.

@testset "credal rules against the Python package improb" begin
    reference = JSON.parsefile(joinpath(VALIDATION, "improb_reference.json"))
    @test reference["provenance"]["improb"] == "0.1.1"
    @test startswith(reference["provenance"]["python"], "2.7")
    @test length(reference["problems"]) >= 33
    # The fixture must be about the same problems as the scipy one.
    finite = JSON.parsefile(joinpath(VALIDATION, "finite_reference.json"))
    @test [(p["name"], p["payoff"], p["credal"]) for p in reference["problems"]] ==
          [(p["name"], p["payoff"], p["credal"]) for p in finite["problems"]]
    separated = (; e_admissible=Ref(0), interval=Ref(0), ties=Ref(0), redundant=Ref(0))
    for ex in reference["problems"]
        Q = fixture_matrix(ex["payoff"])
        points = [fixture_rationals(p) for p in ex["credal"]]
        problem = DecisionProblem(PayoffTable(Q), CredalSet(points))
        actions = action_labels(problem.payoff)

        # Lower and upper expectations: improb's natural extension over the facets.
        lower = aggregate(LowerExpectation(), problem)
        upper = aggregate(UpperExpectation(), problem)
        @test lower == fixture_rationals(ex["lower_expectation"])
        @test upper == fixture_rationals(ex["upper_expectation"])

        # Γ-maximin (`OptLowPrevMaxMin`): every tied action, in table order.
        gamma = decide(problem, GammaMaximin())
        @test gamma.chosen == actions[one_based(ex["gamma_maximin"])]
        # Maximality (`OptLowPrevMax`) and interval dominance (`OptLowPrevMaxInterval`).
        maximal = decide(problem, Maximal()).chosen
        interval = decide(problem, IntervalDominance()).chosen
        @test maximal == actions[Bool.(ex["maximal"])]
        @test interval == actions[Bool.(ex["interval_dominance"])]
        # E-admissibility: improb's sure-loss test on the set cut down to the priors
        # under which the action is optimal (improb has no operator of its own).
        admissible = decide(problem, EAdmissible())
        @test admissible.chosen == actions[Bool.(ex["e_admissible"])]
        @test all(values(admissible.details.exact))

        # improb's extreme points are among our generating points, and dropping the
        # redundant generators changes nothing.
        extreme = [fixture_rationals(p) for p in ex["improb_extreme_points"]]
        @test extreme ⊆ points
        reduced = DecisionProblem(PayoffTable(Q), CredalSet(extreme))
        @test aggregate(LowerExpectation(), reduced) == lower
        @test aggregate(UpperExpectation(), reduced) == upper
        @test decide(reduced, EAdmissible()).chosen == admissible.chosen

        separated.e_admissible[] += admissible.chosen != maximal
        separated.interval[] += maximal != interval
        separated.ties[] += length(gamma.chosen) > 1
        separated.redundant[] += length(extreme) < length(unique(points))
    end
    # The fixture exercises the distinctions between the rules: problems where the
    # E-admissible set is smaller than the maximal one, where interval dominance keeps
    # more than maximality, with a Γ-maximin tie and with a redundant generator.
    @test separated.e_admissible[] >= 3
    @test separated.interval[] >= 8
    @test separated.ties[] >= 1
    @test separated.redundant[] >= 1
end
