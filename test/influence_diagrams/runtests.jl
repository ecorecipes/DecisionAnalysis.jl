# The influence-diagram bridge: `normal_form` against InfluenceDiagrams.jl's own
# solvers, which share no code with this package. Its structural value of perfect
# information and value of an observation must equal `evpi` and `evsi` on the normal
# form, and its max-EU policy must be `decide(problem, MaxExpectedUtility())`.
using Test
using Random
using DecisionAnalysis
using InfluenceDiagrams: InfluenceDiagrams
const ID = InfluenceDiagrams

# A random column-stochastic table as rows of a conditional probability table:
# cpt[parent configuration, child state], rows summing to one.
function random_cpt(rng, parents, child)
    w = rand(rng, parents, child) .+ 0.05
    return w ./ sum(w; dims=2)
end

# Chance X (the state), Y | X (a noisy reading), Z | X, D (a consequence); decision D
# informed by Y; utilities on (Z, D) and on X.
function random_diagram(rng)
    nx, ny, nz, nd = rand(rng, 2:3), rand(rng, 2:3), 2, rand(rng, 2:3)
    label(prefix, n) = [Symbol(prefix, i) for i in 1:n]
    id = ID.influence_diagram(:X => label("x", nx), :Y => label("y", ny),
                              :D => label("d", nd), :Z => label("z", nz);
                              mechanisms=[:Y => :X, :Z => (:X, :D)],
                              decisions=[:D => :Y],
                              utilities=[:Outcome => (:Z, :D), :Standing => :X])
    m = ID.InfluenceDiagramModel(id)
    px = rand(rng, nx) .+ 0.1
    m = ID.bind_cpt(m, [:X => px ./ sum(px), :Y => random_cpt(rng, nx, ny)])
    m = ID.bind_cpt(m, :Z => reshape(random_cpt(rng, nx * nd, nz), nx, nd, nz))
    m = ID.bind_utility(m, :Outcome => 10 .* rand(rng, nz, nd))
    return ID.bind_utility(m, :Standing => 5 .* rand(rng, nx))
end

@testset "influence-diagram bridge" begin
    @test Base.get_extension(DecisionAnalysis, :DecisionAnalysisInfluenceDiagramsExt) !==
          nothing
    # The bridge must not bring Catlab in.
    @test !any(id -> id.name == "Catlab", keys(Base.loaded_modules))

    @testset "the umbrella problem (Shachter 1986)" begin
        m = ID.umbrella_model()
        nf = normal_form(m; states=:Weather)
        @test nf.decision === :Umbrella && nf.states == [:Weather]
        @test nf.information == [:Forecast] && nf.observed == [:Forecast]
        @test action_labels(nf.problem.payoff) == [:take, :leave]
        @test state_labels(nf.problem) == [:sunny, :rainy]
        @test nf.problem.payoff.Q ≈ [20 70; 100 0]
        @test probabilities(nf.problem.belief) ≈ [0.7, 0.3]
        @test nf.experiment.likelihood ≈ [0.7 0.15; 0.2 0.25; 0.1 0.6]
        @test signal_labels(nf.experiment) == [:sunny, :cloudy, :rainy]
        # The published numbers: 70 without information, 77 with the forecast, 91 with
        # the weather.
        @test act_now_value(nf.problem) ≈ 70
        @test evsi(nf.problem, nf.experiment) ≈ 7
        @test evpi(nf.problem) ≈ 21
        # ... and the structural values of the diagram itself.
        blind = ID.without_information(m, :Umbrella, :Forecast)
        @test ID.optimize(blind).expected_utility ≈ act_now_value(nf.problem)
        @test ID.expected_value_of_information(blind, :Forecast, :Umbrella) ≈
              evsi(nf.problem, nf.experiment)
        @test ID.expected_value_of_perfect_information(blind, :Umbrella;
                                                       variables=[:Weather]) ≈
              evpi(nf.problem)
        @test ID.expected_value_of_perfect_information(m, :Umbrella) ≈
              evpi(nf.problem) - evsi(nf.problem, nf.experiment)
        # The max-EU policy: with no information, and signal by signal.
        policy = ID.policies(ID.optimize(blind).strategy)[:Umbrella]
        @test decide(nf.problem, MaxExpectedUtility()).chosen ==
              [only(ID.policy_table(policy))]
        informed = ID.policy_table(ID.policies(ID.optimize(m).strategy)[:Umbrella])
        profile = voi_profile(nf.problem, nf.experiment)
        @test [first(o.best) for o in profile.outcomes] == informed
        # A baseline carries over.
        @test normal_form(m; states=:Weather, baseline=:leave).problem.baseline === :leave
        # The default state is every chance variable upstream of the decision: the
        # forecast is then part of the state and there is nothing left to observe.
        full = normal_form(m)
        @test full.states == [:Weather, :Forecast] && full.experiment === nothing
        @test evpi(full.problem) ≈ 21
        @test evppi(full.problem, repeat([:sunny, :cloudy, :rainy]; inner=2)) ≈ 7
    end

    @testset "random single-decision diagrams" begin
        rng = Xoshiro(91)
        for _ in 1:25
            m = random_diagram(rng)
            blind = ID.without_information(m, :D, :Y)
            base = ID.optimize(blind)
            nf = normal_form(m; states=:X)
            problem, experiment = nf.problem, nf.experiment
            @test act_now_value(problem) ≈ base.expected_utility
            # Value of the observation the decision has, and of the whole state.
            @test evsi(problem, experiment) ≈
                  ID.expected_value_of_information(blind, :Y, :D) atol = 1e-9
            @test evpi(problem) ≈
                  ID.expected_value_of_perfect_information(blind, :D; variables=[:X]) atol = 1e-9
            @test value_with_information(problem, experiment) ≈
                  ID.optimize(m).expected_utility
            # The same by brute force over every policy.
            @test value_with_information(problem, experiment) ≈
                  ID.optimize(m, ID.ExhaustivePolicySearch()).expected_utility
            # The max-EU action of the diagram is a Bayes action of the normal form, and
            # its expected utility is that action's score.
            action = only(ID.policy_table(ID.policies(base.strategy)[:D]))
            result = decide(problem, MaxExpectedUtility())
            @test action in result.chosen
            @test ID.expected_utility(blind, :D => action) ≈ maximum(result.scores)
            for (a, score) in zip(action_labels(problem.payoff), result.scores)
                @test ID.expected_utility(blind, :D => a) ≈ score
            end
            # Signal by signal, the diagram's policy is optimal after that signal.
            informed = ID.policy_table(ID.policies(ID.optimize(m).strategy)[:D])
            for (o, a) in zip(voi_profile(problem, experiment).outcomes, informed)
                @test a in o.best
            end
            # With the reading in the state as well, observing it is a partition.
            full = normal_form(m)
            @test full.states == [:X, :Y] && full.experiment === nothing
            @test evpi(full.problem) ≈ evpi(problem) atol = 1e-9
            ny = length(signal_labels(experiment))
            nx = length(state_labels(problem))
            @test evppi(full.problem, repeat(1:ny; inner=nx)) ≈
                  evsi(problem, experiment) atol = 1e-9
        end
    end

    @testset "what the bridge refuses" begin
        m = ID.umbrella_model()
        # The forecast alone is not a state: the utility depends on the weather.
        @test evpi(normal_form(m; states=:Forecast).problem) ≈ 7
        @test_throws ArgumentError normal_form(m; states=:Umbrella)
        @test_throws ArgumentError normal_form(m; states=:Nothing)
        @test_throws ArgumentError normal_form(ID.observe(m, :Forecast => :rainy);
                                               states=:Weather)
        @test_throws ArgumentError normal_form(ID.two_stage_model())
        # A state that leaves out what the utility depends on, while the decision
        # observes it: the experiment would not be a likelihood for that state alone.
        rng = Xoshiro(92)
        id = ID.influence_diagram(:X => [:a, :b], :W => [:u, :v], :D => [:p, :q];
                                  decisions=[:D => :W], utilities=[:U => (:X, :W, :D)])
        model = ID.InfluenceDiagramModel(id)
        model = ID.bind_cpt(model, [:X => [0.4, 0.6], :W => [0.5, 0.5]])
        model = ID.bind_utility(model, :U => rand(rng, 2, 2, 2))
        @test_throws ArgumentError normal_form(model; states=:X)
        @test normal_form(model; states=[:X, :W]).experiment === nothing
    end
end

# The Bayesian-network bridge: a decision problem as a set of networks, one per action.
using BayesianNetworks: BayesianNetworks
const BN = BayesianNetworks

# The pond of the tutorials as a network with a management node: Bd, two tests of it, the
# response, and a count that depends on Bd and the response.
function pond_network()
    states = [:absent, :low_prevalence, :high_prevalence]
    actions = [:no_intervention, :antifungal_baths, :disinfect_pond]
    bn = BN.bayesnet(:Bd => states, :Swab => [:negative, :positive],
                     :Visual => [:no_signs, :signs], :Response => actions;
                     mechanisms=[:Swab => (:Bd,), :Visual => (:Bd,)])
    m = BN.bind_cpt(BN.BayesModel(bn),
                    [:Bd => [0.5, 0.3, 0.2], :Swab => [0.95 0.05; 0.4 0.6; 0.05 0.95],
                     :Visual => [0.55 0.45; 0.5 0.5; 0.45 0.55],
                     :Response => [1 / 3, 1 / 3, 1 / 3]])
    frogs = [500 300 100; 470 380 250; 350 340 320] .- [0 150 250; 0 60 120; 0 10 20]
    return m, permutedims(frogs)               # (Bd, Response)
end

@testset "Bayesian-network bridge" begin
    @test Base.get_extension(DecisionAnalysis, :DecisionAnalysisBayesianNetworksExt) !==
          nothing

    @testset "one network with a management node" begin
        m, frogs = pond_network()
        nf = normal_form(m; decision=:Response, states=:Bd, observed=(:Swab, :Visual),
                         outcome=(:Bd, :Response) => frogs, baseline=:no_intervention)
        @test nf.decision === :Response && nf.states == [:Bd]
        @test nf.actions == [:no_intervention, :antifungal_baths, :disinfect_pond]
        @test nf.observed == [:Swab, :Visual]
        problem = nf.problem
        @test problem.payoff.Q ≈ permutedims(frogs)
        @test probabilities(problem.belief) ≈ [0.5, 0.3, 0.2]
        @test problem.baseline === :no_intervention
        # The numbers of the tutorials on this pond, which were computed exactly.
        @test evpi(problem) ≈ 52
        @test evsi(problem, nf.experiments.Swab) ≈ 311 / 10
        @test nf.experiments.Swab.likelihood ≈ [0.95 0.4 0.05; 0.05 0.6 0.95]
        @test nf.experiments.Visual.likelihood ≈ [0.55 0.5 0.45; 0.45 0.5 0.55]
        @test signal_labels(nf.experiments.Swab) == [:negative, :positive]
        # Both tests together: four signals, the first variable fastest, and worth at
        # least as much as either alone.
        @test signal_labels(nf.experiment) ==
              [:negative_no_signs, :positive_no_signs, :negative_signs, :positive_signs]
        both = evsi(problem, nf.experiment)
        @test both >= evsi(problem, nf.experiments.Swab) - 1e-9
        @test both >= evsi(problem, nf.experiments.Visual) - 1e-9
        @test both <= evpi(problem) + 1e-9
        # The same problem from the influence diagram, which shares no code with this.
        id = ID.influence_diagram(:Bd => BN.states(BN.syntax(m), :Bd),
                                  :Swab => [:negative, :positive],
                                  :Response => nf.actions;
                                  mechanisms=[:Swab => :Bd], decisions=[:Response => :Swab],
                                  utilities=[:Frogs => (:Bd, :Response)])
        diagram = ID.bind_cpt(ID.InfluenceDiagramModel(id),
                              [:Bd => [0.5, 0.3, 0.2],
                               :Swab => [0.95 0.05; 0.4 0.6; 0.05 0.95]])
        diagram = ID.bind_utility(diagram, :Frogs => Float64.(frogs))
        from_diagram = normal_form(diagram; states=:Bd)
        @test from_diagram.problem.payoff.Q ≈ problem.payoff.Q
        @test from_diagram.experiment.likelihood ≈ nf.experiments.Swab.likelihood
        @test ID.optimize(diagram).expected_utility ≈
              value_with_information(problem, nf.experiments.Swab)
        # No observation, no experiment.
        plain = normal_form(m; decision=:Response, states=:Bd,
                            outcome=(:Bd, :Response) => frogs)
        @test plain.experiment === nothing && isempty(plain.experiments)
        @test plain.problem.payoff.Q ≈ problem.payoff.Q
    end

    @testset "a named collection of networks" begin
        m, frogs = pond_network()
        actions = BN.states(BN.syntax(m), :Response)
        models = NamedTuple{Tuple(actions)}(Tuple(BN.do_intervention(m, :Response => a)
                                                  for a in actions))
        from_one = normal_form(m; decision=:Response, states=:Bd, observed=:Swab,
                               outcome=(:Bd, :Response) => frogs)
        from_many = normal_form(models; states=:Bd, observed=:Swab,
                                outcome=(:Bd, :Response) => frogs)
        @test from_many.problem.payoff.Q == from_one.problem.payoff.Q
        @test from_many.experiment.likelihood == from_one.experiment.likelihood
        @test from_many.actions == actions
        # B23: Dict and Vector{Pair} collections of BayesModels also dispatch cleanly.
        from_pairs = normal_form([a => BN.do_intervention(m, :Response => a)
                                  for a in actions];
                                 states=:Bd, observed=:Swab,
                                 outcome=(:Bd, :Response) => frogs)
        @test from_pairs.problem.payoff.Q == from_one.problem.payoff.Q
        from_dict = normal_form(Dict(a => BN.do_intervention(m, :Response => a)
                                     for a in actions);
                                states=:Bd, observed=:Swab,
                                outcome=(:Bd, :Response) => frogs)
        @test Set(action_labels(from_dict.problem)) == Set(actions)
        # A count read from an outcome node, by state name, by vector, and as two terms.
        habitat = BN.reference_habitat_model()
        by_name = normal_form(habitat; decision=:Irrigation, states=:Climate,
                              outcome=:Occupancy => (absent=0, present=100), baseline=:low)
        by_vector = normal_form(habitat; decision=:Irrigation, states=:Climate,
                                outcome=:Occupancy => [0, 100])
        @test by_name.problem.payoff.Q == by_vector.problem.payoff.Q
        @test by_name.problem.baseline === :low
        for (a, action) in enumerate((:low, :high)),
            (s, climate) in enumerate((:dry, :normal, :wet))
            conditioned = BN.do_intervention(habitat,
                                             [:Irrigation => action, :Climate => climate])
            @test by_name.problem.payoff.Q[a, s] ≈
                  100 * BN.marginal(conditioned, :Occupancy).table[2]
        end
        with_term = normal_form(habitat; decision=:Irrigation, states=:Climate,
                                outcome=[:Occupancy => [0, 100],
                                         :Irrigation => (low=0, high=-5)])
        @test with_term.problem.payoff.Q ≈ by_name.problem.payoff.Q .+ [0; -5]
        # Two states: their joint, the first fastest.
        two = normal_form(habitat; decision=:Irrigation,
                          states=(:Climate, :GrazingPressure),
                          outcome=:Occupancy => [0, 100])
        @test state_labels(two.problem) ==
              [:dry_low, :normal_low, :wet_low, :dry_high, :normal_high, :wet_high]
        @test sum(probabilities(two.problem.belief)) ≈ 1
        @test aggregate(Expectation(), two.problem) ≈
              aggregate(Expectation(), by_name.problem)
    end

    @testset "hypotheses" begin
        habitat = BN.reference_habitat_model()
        doubtful = BN.bind_cpt(habitat, :Occupancy => [0.6 0.4; 0.5 0.5])
        networks(model) = (; low=BN.do_intervention(model, :Irrigation => :low),
                           high=BN.do_intervention(model, :Irrigation => :high))
        outcome = [:Occupancy => [0, 100], :Irrigation => (low=0, high=-5)]
        nf = normal_form((; published=networks(habitat), doubtful=networks(doubtful));
                         outcome, baseline=:low)
        @test nf.actions == [:low, :high] &&
              state_labels(nf.problem) == [:published, :doubtful]
        @test nf.problem.belief isa ScenarioSet
        @test nf.experiment === nothing
        for (h, model) in enumerate((habitat, doubtful)),
            (a, action) in enumerate((:low, :high))
            fixed = BN.do_intervention(model, :Irrigation => action)
            @test nf.problem.payoff.Q[a, h] ≈
                  100 * BN.marginal(fixed, :Occupancy).table[2] - 5 * (a == 2)
        end
        # With weights on the hypotheses there is an expectation, and a value of knowing
        # which network is right.
        weighted = normal_form((; published=networks(habitat), doubtful=networks(doubtful));
                               outcome, prior=[3 // 4, 1 // 4], baseline=:low)
        @test weighted.problem.belief isa FiniteBelief
        @test probabilities(weighted.problem.belief) == [3 // 4, 1 // 4]
        @test evpi(weighted.problem) >= 0
        @test_throws ArgumentError normal_form((; a=networks(habitat),
                                                b=(; low=habitat, other=habitat)); outcome)
    end

    @testset "what the bridge refuses" begin
        m, frogs = pond_network()
        outcome = (:Bd, :Response) => frogs
        habitat = BN.reference_habitat_model()
        count = :Occupancy => [0, 100]
        # Something downstream of the decision is not observed before it, and is no state.
        @test_throws ArgumentError normal_form(habitat; decision=:Irrigation,
                                               states=:Climate,
                                               observed=:SoilMoisture, outcome=count)
        @test_throws ArgumentError normal_form(habitat; decision=:Irrigation,
                                               states=:SoilMoisture, outcome=count)
        # A state that leaves out what the outcome depends on, while it is observed.
        @test_throws ArgumentError normal_form(m; decision=:Response, states=:Visual,
                                               observed=:Swab, outcome)
        @test_throws ArgumentError normal_form(m; decision=:Response, states=:Response,
                                               outcome)
        @test_throws ArgumentError normal_form(m; decision=:Response, states=:Bd,
                                               observed=:Bd, outcome)
        @test_throws ArgumentError normal_form(m; decision=:Nowhere, states=:Bd, outcome)
        @test_throws ArgumentError normal_form(m; decision=:Response, states=:Nowhere,
                                               outcome)
        @test_throws ArgumentError normal_form(m; decision=:Response, states=:Bd,
                                               outcome=(:Bd, :Response) => frogs[:, 1:2])
        @test_throws ArgumentError normal_form(habitat; decision=:Irrigation,
                                               states=:Climate,
                                               outcome=:Occupancy => (absent=0, here=1))
        @test_throws ArgumentError normal_form(habitat; decision=:Irrigation,
                                               states=:Climate,
                                               outcome=(:Occupancy, :Climate) => (a=1,))
        @test_throws ArgumentError normal_form(habitat; decision=:Irrigation, states=(),
                                               outcome=count)
        @test_throws InvalidBaselineError normal_form(habitat; decision=:Irrigation,
                                                      states=:Climate, outcome=count,
                                                      baseline=:none)
    end
end
