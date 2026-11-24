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
