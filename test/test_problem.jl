@testset "payoff tables and problems" begin
    Q = [0 -10 -60 -150; -20 -20 -25 -90; -48 -48 -48 -48]
    t = PayoffTable([:nothing, :levee, :relocate], [:low, :moderate, :high, :extreme], Q)

    @testset "PayoffTable" begin
        @test size(t) == (3, 4)
        @test eltype(t) == Int
        @test action_labels(t) == [:nothing, :levee, :relocate]
        @test state_labels(t) == [:low, :moderate, :high, :extreme]
        default = PayoffTable(Q)
        @test action_labels(default) == [:a1, :a2, :a3]
        @test state_labels(default) == [:s1, :s2, :s3, :s4]
        # The matrix is copied.
        M = [1.0 2.0; 3.0 4.0]
        copied = PayoffTable(M)
        M[1, 1] = 99.0
        @test copied.Q[1, 1] == 1.0
        @test eltype(PayoffTable(BigRat.(Q))) == BigRat
        se = fill(0.1, 3, 4)
        @test PayoffTable(Q; standard_errors=se).standard_errors == se
        @test t.standard_errors === nothing
        @test_throws InvalidPayoffError PayoffTable([:a], [:s, :t], Q)
        @test_throws InvalidPayoffError PayoffTable([:a, :a, :b], [:p, :q, :r, :s], Q)
        @test_throws InvalidPayoffError PayoffTable([1.0 Inf; 0.0 1.0])
        @test_throws InvalidPayoffError PayoffTable([1.0 NaN; 0.0 1.0])
        @test_throws InvalidPayoffError PayoffTable(zeros(0, 2))
        @test_throws InvalidPayoffError PayoffTable(Q; standard_errors=fill(0.1, 2, 2))
        @test_throws InvalidPayoffError PayoffTable(Q; standard_errors=fill(-0.1, 3, 4))
    end

    @testset "DecisionProblem" begin
        p = [1 // 2, 3 // 10, 3 // 20, 1 // 20]
        problem = DecisionProblem(t, p)
        @test problem.belief == FiniteBelief(t.states, p)
        @test feasible_actions(problem) == action_labels(problem) == t.actions
        @test state_labels(problem) == t.states
        @test_throws LabelMismatchError DecisionProblem(t, FiniteBelief(p))
        @test_throws InvalidBeliefError DecisionProblem(t, [1 // 2, 1 // 2])
        @test_throws LabelMismatchError DecisionProblem(t, PointBelief(:tsunami))
        @test_throws LabelMismatchError DecisionProblem(t, ScenarioSet([:low, :tsunami]))
        @test DecisionProblem(t, ScenarioSet([:low, :extreme])) isa DecisionProblem
    end

    @testset "feasibility filter" begin
        p = [1 // 2, 3 // 10, 3 // 20, 1 // 20]
        full = DecisionProblem(t, p)
        for filter in ([true, false, true], [:nothing, :relocate], ["nothing", "relocate"],
                       a -> a != :levee)
            restricted = DecisionProblem(t, p; feasible=filter)
            @test feasible_actions(restricted) == [:nothing, :relocate]
            @test feasible_payoff(restricted).Q == Q[[1, 3], :]
            # The filtered problem equals the problem built from the smaller table.
            small = DecisionProblem(PayoffTable([:nothing, :relocate], t.states,
                                                Q[[1, 3], :]), p)
            @test aggregate(Expectation(), restricted) == aggregate(Expectation(), small)
            @test regret_table(restricted) == regret_table(small)
            @test max_regret(restricted) == max_regret(small)
            @test evpi(restricted) == evpi(small)
            @test decide(restricted, MinimaxRegret()).chosen ==
                  decide(small, MinimaxRegret()).chosen
        end
        # Dropping an action changes regret: a different answer from the full problem.
        @test regret_table(DecisionProblem(t, p; feasible=[:nothing, :relocate])) !=
              regret_table(full)[[1, 3], :]
        @test_throws NoFeasibleActionError DecisionProblem(t, p;
                                                           feasible=[false, false,
                                                                     false])
        @test_throws ArgumentError DecisionProblem(t, p; feasible=[:swim])
        @test_throws ArgumentError DecisionProblem(t, p; feasible=[true, false])
    end

    @testset "restrict_scenarios" begin
        p = [1 // 2, 3 // 10, 3 // 20, 1 // 20]
        r = restrict_scenarios(DecisionProblem(t, p), [:extreme, :low])
        @test state_labels(r) == [:low, :extreme]            # table order
        @test probabilities(r.belief) == [10 // 11, 1 // 11]  # conditioned
        @test r.payoff.Q == Q[:, [1, 4]]
        c = CredalSet(t.states, [p, [7 // 20, 3 // 10, 1 // 4, 1 // 10]])
        rc = restrict_scenarios(DecisionProblem(t, c), [:low, :extreme])
        @test extreme_points(rc.belief) == [[10 // 11, 1 // 11], [7 // 9, 2 // 9]]
        rs = restrict_scenarios(DecisionProblem(t, ScenarioSet([:low, :high])),
                                [:high, :extreme])
        @test state_labels(rs.belief) == [:high]
        @test restrict_scenarios(DecisionProblem(t, PointBelief(:high)), [:high]).belief ==
              PointBelief(:high)
        @test_throws ArgumentError restrict_scenarios(DecisionProblem(t,
                                                                      PointBelief(:high)),
                                                      [:low])
        @test_throws ArgumentError restrict_scenarios(DecisionProblem(t, p), [:tsunami])
        @test_throws ArgumentError restrict_scenarios(DecisionProblem(t, p), Symbol[])
        zero_mass = DecisionProblem(t, [1 // 1, 0 // 1, 0 // 1, 0 // 1])
        @test_throws ArgumentError restrict_scenarios(zero_mass, [:high])
        @test_throws ArgumentError restrict_scenarios(DecisionProblem(t,
                                                                      ScenarioSet([:low])),
                                                      [:high])
    end

    @testset "random tables" begin
        a = random_payoff_table(Random.Xoshiro(7), 3, 4; T=BigRat)
        @test a == random_payoff_table(Random.Xoshiro(7), 3, 4; T=BigRat)
        @test size(random_payoff_table(Random.Xoshiro(7), 2, 5)) == (2, 5)
    end
end

@testset "problems are compared by value" begin
    table = PayoffTable([:a, :b], [:x, :y], [1 0; 0 1])
    make() = DecisionProblem(table, [1 // 2, 1 // 2]; baseline=:a)
    @test make() == make() && hash(make()) == hash(make())
    @test make() != DecisionProblem(table, [1 // 3, 2 // 3]; baseline=:a)
    @test make() != DecisionProblem(table, [1 // 2, 1 // 2])
    @test make() != DecisionProblem(table, [1 // 2, 1 // 2]; baseline=:a, feasible=[:a])
    @test length(unique([make(), make()])) == 1
end
