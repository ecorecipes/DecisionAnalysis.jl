# Scenario designs (Runge et al. 2024): axes that are decisions or uncertainties, the
# class and purpose of a design, the runs it plans, fractions, effects of the axes,
# forecasts from weighted scenarios, and whether the better action depends on the state.
# Oracles: explicit loops and the analyses of the rest of the package.

# A model that reads its scenario: the adapter verbs on a marker type.
struct ScenarioToy end
DecisionAnalysis.with_parameters(::ScenarioToy, θ) = (; θ...)
DecisionAnalysis.apply_action(p::NamedTuple, a::NamedTuple) = merge(p, a)
DecisionAnalysis.simulate(p::NamedTuple; rng) = merge(p, (; noise=rand(rng)))
function DecisionAnalysis.outcome(p::NamedTuple, run::NamedTuple)
    boost = get(p, :boosters, :none)
    escape = get(p, :variant, 1.0)
    averted = boost === :none ? 0.0 : boost === :older ? 10.0 : 8.0 * escape
    return (; averted=averted + run.noise, first=run.noise)
end

@testset "scenario designs" begin
    boosters = decision_axis(:boosters, [:none, :older, :all])
    variant = uncertainty_axis(:variant, (; none=1.0, escape=1.4))
    design = ScenarioDesign(boosters, variant; purpose=:value_of_information,
                            audience="an advisory committee")

    @testset "axes, class and purpose" begin
        @test boosters.kind === :decision && variant.kind === :uncertainty
        @test variant.levels == [:none, :escape] && variant.values == Any[1.0, 1.4]
        @test boosters.values == Any[:none, :older, :all]
        @test design_class(design) === :value_of_information
        @test design_class(ScenarioDesign(boosters)) === :decision
        @test design_class(ScenarioDesign(variant)) === :uncertainty
        @test design.purpose === :value_of_information &&
              design.audience == "an advisory committee"
        # Every purpose with the axes it needs, and refused with the others.
        for purpose in (:sensitivity_analysis, :situational_awareness, :horizon_scanning)
            @test ScenarioDesign(variant; purpose).purpose === purpose
            @test_throws ArgumentError ScenarioDesign(boosters, variant; purpose)
        end
        @test ScenarioDesign(boosters; purpose=:decision_making).purpose ===
              :decision_making
        @test_throws ArgumentError ScenarioDesign(boosters, variant;
                                                  purpose=:decision_making)
        @test_throws ArgumentError ScenarioDesign(boosters; purpose=:value_of_information)
        @test_throws ArgumentError ScenarioDesign(variant; purpose=:value_of_information)
        @test_throws ArgumentError ScenarioDesign(variant; purpose=:prediction)
        # A forecast needs probabilities of every level.
        @test_throws ArgumentError ScenarioDesign(variant; purpose=:forecasting)
        weighted = uncertainty_axis(:variant, (; none=1.0, escape=1.4); weights=[3, 1])
        @test weighted.weights == [3 // 4, 1 // 4]
        @test ScenarioDesign(weighted; purpose=:forecasting).purpose === :forecasting
        @test_throws ArgumentError ScenarioAxis(:x, [:a, :b]; kind=:factor)
        @test_throws ArgumentError ScenarioAxis(:x, Symbol[]; kind=:decision)
        @test_throws ArgumentError ScenarioAxis(:x, [:a, :a]; kind=:decision)
        @test_throws ArgumentError ScenarioAxis(:x, [:a, :b]; kind=:decision, weights=[1, 1])
        @test_throws ArgumentError uncertainty_axis(:x, [:a, :b]; weights=[1])
        @test_throws ArgumentError ScenarioDesign(boosters, boosters)
        @test_throws ArgumentError ScenarioDesign()
    end

    @testset "the runs a design plans" begin
        rows = scenarios(design)
        @test length(rows) == 6 && Tables.istable(rows)
        @test rows[1] == (; scenario=:none_none, boosters=:none, variant=:none)
        @test rows[6] == (; scenario=:all_escape, boosters=:all, variant=:escape)
        @test design_actions(design) == [:none => (; boosters=:none),
                                         :older => (; boosters=:older),
                                         :all => (; boosters=:all)]
        draws = design_draws(design)
        @test state_labels(draws) == [:none, :escape]
        @test parameter_draws(draws) == (; variant=[1.0, 1.4])
        @test design_belief(design) isa ScenarioSet
        @test design_belief(design).states == [:none, :escape]
        # Without one kind of axis.
        @test design_actions(ScenarioDesign(variant)) == [:projection => (;)]
        @test state_labels(design_draws(ScenarioDesign(boosters))) == [:all]
        @test design_belief(ScenarioDesign(boosters)) isa FiniteBelief
        # Weights multiply across axes.
        two = ScenarioDesign(uncertainty_axis(:a, [:lo, :hi]; weights=[1, 3]),
                             uncertainty_axis(:b, [:x, :y, :z]; weights=[1, 1, 2]))
        belief = design_belief(two)
        @test belief isa FiniteBelief && sum(belief.probabilities) == 1
        @test belief.probabilities[findfirst(==(:hi_z), belief.states)] == 3 // 8
        @test draw_weights(design_draws(two)) == belief.probabilities

        # Run it, paired across scenarios, and analyse it over the scenario set.
        table = outcomes(ScenarioToy(), design_actions(design), draws; replicates=4, seed=3)
        @test action_labels(table) == [:none, :older, :all]
        @test table.values[1, :, :, 2] == table.values[3, :, :, 2]     # paired replicates
        problem = DecisionProblem(design, table, IndividualsUtility(:averted);
                                  baseline=:none)
        @test problem.belief isa ScenarioSet
        gains = gain_over_baseline(problem).Q
        @test gains ≈ [0 0; 10 10; 8 11.2]
        @test_throws BeliefFormError evpi(problem)
        @test decide(problem, MinimaxRegret()).chosen == [:older]
        robust = decision_robustness(problem)
        @test robust.reversal && isempty(robust.robust)
        @test robust.best == [:none => [:older], :escape => [:all]]
        @test Dict(robust.losses)[:older] ≈ 1.2 && Dict(robust.losses)[:all] ≈ 2
        # With probabilities the same runs have an expected value of information.
        likely = ScenarioDesign(boosters,
                                uncertainty_axis(:variant, (; none=1.0, escape=1.4);
                                                 weights=[3, 1]))
        valued = DecisionProblem(likely, table, IndividualsUtility(:averted);
                                 baseline=:none)
        @test valued.belief isa FiniteBelief
        @test evpi(valued) ≈ 0.25 * 1.2
        # No reversal: one action best everywhere, and nothing to learn.
        calm = DecisionProblem(PayoffTable([0 0; 3 2; 1 1]; actions=[:none, :a, :b]),
                               ScenarioSet([:s1, :s2]))
        steady = decision_robustness(calm)
        @test !steady.reversal && steady.robust == [:a]
        @test Dict(steady.losses) == Dict(:none => 3, :a => 0, :b => 2)
        # A tie in one state does not make a reversal.
        @test !decision_robustness(DecisionProblem(PayoffTable([1 2; 1 1]),
                                                   ScenarioSet([:s1, :s2]))).reversal
        # The table must be the design's.
        other = outcomes(ScenarioToy(), [:only => (; boosters=:none)], draws; seed=1)
        @test_throws ArgumentError DecisionProblem(design, other,
                                                   IndividualsUtility(:averted))
    end

    @testset "effects of the axes and fractions" begin
        r0 = uncertainty_axis(:r0, (; low=2.2, high=3.0); weights=[3, 2])
        serial = uncertainty_axis(:serial, (; short=3.0, long=4.5); weights=[1, 1])
        severity = uncertainty_axis(:severity, [:mild, :severe]; weights=[3, 1])
        full = ScenarioDesign(r0, serial, severity; purpose=:sensitivity_analysis)
        output(s) = 10.0 * (s.r0 === :high ? 2 : 1) + (s.serial === :long ? 3 : 0) +
                    (s.r0 === :high && s.severity === :severe ? 4 : 0)
        values = output.(scenarios(full))
        effects = factor_effects(full, values)
        main = Dict(m.axis => m for m in effects.main)
        @test effects.grand_mean ≈ sum(values) / 8
        @test main[:r0].effect ≈ 12 && main[:serial].effect ≈ 3 &&
              main[:severity].effect ≈ 2
        @test main[:r0].means ≈ [11.5, 23.5] && main[:r0].kind === :uncertainty
        # The sums of squares add up: three main effects and one interaction.
        interactions = Dict(i.axes => i.share for i in effects.interactions)
        @test interactions[(:r0, :severity)] > 0
        @test interactions[(:r0, :serial)] ≈ 0 atol = 1e-12
        @test sum(m.share for m in effects.main) + sum(Base.values(interactions)) ≈ 1
        # Shares against explicit sums of squares.
        total = sum(abs2, values .- sum(values) / 8)
        @test main[:serial].share ≈ 8 * 1.5^2 / total
        # Keyed by scenario label, in any order.
        keyed = Dict(s.scenario => output(s) for s in reverse(scenarios(full)))
        @test factor_effects(full, keyed).main == effects.main
        @test_throws ArgumentError factor_effects(full, values[1:5])
        @test_throws ArgumentError factor_effects(full, Dict(:low_short_mild => 1.0))

        # Half of it: severity tied to r0 × serial. Main effects are still estimated,
        # but severity is aliased with the interaction of the other two.
        half = fractional_design(full, :severity => (:r0, :serial))
        kept = scenarios(half)
        @test length(kept) == 4 && half.purpose === :sensitivity_analysis
        sign(s, axis, high) = getproperty(s, axis) === high ? 1 : -1
        @test all(sign(s, :severity, :severe) == sign(s, :r0, :high) *
                                                 sign(s, :serial, :long) for s in kept)
        additive(s) = 5.0 * sign(s, :r0, :high) + 2.0 * sign(s, :serial, :long) +
                      1.0 * sign(s, :severity, :severe)
        fraction = factor_effects(half, additive.(kept))
        @test [m.effect for m in fraction.main] ≈ [10, 4, 2]     # twice each coefficient
        @test isempty(fraction.interactions)
        aliased(s) = 3.0 * sign(s, :r0, :high) * sign(s, :serial, :long)
        @test Dict(m.axis => m.effect for m in factor_effects(half, aliased.(kept)).main)[:severity] ≈
              6
        @test length(scenarios(fractional_design(full, :severity => :r0))) == 4
        @test_throws ArgumentError fractional_design(full, :severity => (:severity, :r0))
        @test_throws ArgumentError fractional_design(full, :nothing => (:r0,))
        @test_throws ArgumentError fractional_design(ScenarioDesign(decision_axis(:b,
                                                                                  [:x, :y, :z])),
                                                     :b => (:b,))
        @test_throws ArgumentError DecisionProblem(fractional_design(ScenarioDesign(decision_axis(:boosters,
                                                                                                  [:none,
                                                                                                   :all]),
                                                                                    uncertainty_axis(:variant,
                                                                                                     (;
                                                                                                      none=1.0,
                                                                                                      escape=1.4))),
                                                                     :variant => :boosters),
                                                   outcomes(ScenarioToy(),
                                                            [:none => (; boosters=:none),
                                                             :all => (; boosters=:all)],
                                                            SampleBelief((; variant=[1.0, 1.4]);
                                                                         labels=[:none,
                                                                                 :escape]);
                                                            seed=1),
                                                   IndividualsUtility(:averted))

        # A forecast is the weighted mixture of the scenarios.
        forecasting = ScenarioDesign(r0, serial, severity; purpose=:forecasting)
        w = Float64.(design_belief(forecasting).probabilities)
        forecast = scenario_forecast(forecasting, values)
        @test forecast.mean ≈ sum(w .* values)
        @test forecast.between ≈ sum(w .* (values .- forecast.mean) .^ 2)
        @test iszero(forecast.within) && forecast.quantiles === nothing
        @test collect(Base.values(forecast.weights)) ≈ w
        spread = [v .+ [-1.0, 0.0, 1.0] for v in values]
        replicated = scenario_forecast(forecasting, spread; probabilities=[0.0, 0.5, 1.0])
        @test replicated.mean ≈ forecast.mean && replicated.within ≈ 2 / 3
        @test replicated.variance ≈ replicated.between + replicated.within
        @test replicated.quantiles[1] == minimum(values) - 1 &&
              replicated.quantiles[3] == maximum(values) + 1
        @test minimum(values) <= replicated.quantiles[2] <= maximum(values)
        # Scenarios without probabilities are projections, not a forecast; scenarios
        # that differ in a decision are compared, not combined.
        @test_throws ArgumentError scenario_forecast(ScenarioDesign(uncertainty_axis(:r0,
                                                                                     [:low,
                                                                                      :high])),
                                                     [1.0, 2.0])
        @test_throws ArgumentError scenario_forecast(ScenarioDesign(decision_axis(:b,
                                                                                  [:x, :y])),
                                                     [1.0, 2.0])
        @test_throws ArgumentError scenario_forecast(forecasting, vcat(values[1:7], "x"))
    end
end
