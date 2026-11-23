@testset "beliefs" begin
    @testset "FiniteBelief" begin
        b = FiniteBelief([:low, :high], [3 // 5, 2 // 5])
        @test state_labels(b) == [:low, :high]
        @test probabilities(b) == [3 // 5, 2 // 5]
        @test FiniteBelief(["low", "high"], [3 // 5, 2 // 5]) == b
        @test state_labels(FiniteBelief([0.25, 0.75])) == [:s1, :s2]
        @test FiniteBelief([0.1, 0.2, 0.7]) isa FiniteBelief{Float64}
        # Exact arithmetic has no tolerance; floating point has a small one.
        @test_throws InvalidBeliefError FiniteBelief([1 // 2, 1 // 3])
        @test_throws InvalidBeliefError FiniteBelief([0.5, 0.6])
        @test_throws InvalidBeliefError FiniteBelief([1.5, -0.5])
        @test_throws InvalidBeliefError FiniteBelief([NaN, 1.0])
        @test_throws InvalidBeliefError FiniteBelief(Float64[])
        @test_throws InvalidBeliefError FiniteBelief([:a, :a], [0.5, 0.5])
        @test_throws InvalidBeliefError FiniteBelief([:a], [0.5, 0.5])
        @test FiniteBelief([0.5, 0.5 + 1e-10]) isa FiniteBelief
    end

    @testset "from a FiniteKernels state" begin
        X = FiniteKernels.FiniteSpace(:Flood, [:low, :high])
        b = FiniteBelief(FiniteKernels.state(X, [0.25, 0.75]))
        @test state_labels(b) == [:low, :high]
        @test probabilities(b) == [0.25, 0.75]
        exact = FiniteBelief(FiniteKernels.state(X, [1 // 4, 3 // 4]))
        @test exact isa FiniteBelief{Rational{Int}}
        # A joint space: labels joined in joint_states order, matching the table.
        XY = FiniteKernels.tensor_space(X, FiniteKernels.FiniteSpace(:Wind, [:calm, :gale]))
        joint = FiniteKernels.state(XY, [0.1 0.2; 0.3 0.4])
        bj = FiniteBelief(joint)
        @test length(state_labels(bj)) == 4
        for (i, s) in enumerate(FiniteKernels.joint_states(XY))
            @test state_labels(bj)[i] == Symbol(join(s, "_"))
            @test probabilities(bj)[i] == FiniteKernels.probability(joint, s)
        end
        # A kernel with a domain is not a state.
        @test_throws ArgumentError FiniteBelief(FiniteKernels.identity_kernel(X))
    end

    @testset "CredalSet, ScenarioSet, PointBelief" begin
        c = CredalSet([:low, :high], [[1 // 2, 1 // 2], [1 // 4, 3 // 4]])
        @test extreme_points(c) == [[1 // 2, 1 // 2], [1 // 4, 3 // 4]]
        @test state_labels(c) == [:low, :high]
        members = [FiniteBelief([:low, :high], q) for q in extreme_points(c)]
        @test CredalSet(members) == c
        @test state_labels(CredalSet([[0.5, 0.5]])) == [:s1, :s2]
        @test_throws InvalidBeliefError CredalSet([:a, :b], Vector{Float64}[])
        @test_throws InvalidBeliefError CredalSet([:a, :b], [[0.5, 0.5], [0.5, 0.6]])
        @test_throws InvalidBeliefError CredalSet([:a, :b], [[1.0]])
        @test_throws LabelMismatchError CredalSet([FiniteBelief([:a, :b], [0.5, 0.5]),
                                                   FiniteBelief([:a, :c], [0.5, 0.5])])
        @test state_labels(ScenarioSet(["a", "b"])) == [:a, :b]
        @test_throws InvalidBeliefError ScenarioSet(Symbol[])
        @test_throws InvalidBeliefError ScenarioSet([:a, :a])
        @test state_labels(PointBelief("high")) == [:high]
    end

    @testset "random generators are seeded and valid" begin
        a = random_belief(Random.Xoshiro(1), 5; T=BigRat)
        @test a == random_belief(Random.Xoshiro(1), 5; T=BigRat)
        @test sum(probabilities(a)) == 1
        @test all(>(0), probabilities(random_belief(Random.Xoshiro(2), 4)))
        c = random_credal_set(Random.Xoshiro(3), 4, 3; T=BigRat)
        @test length(extreme_points(c)) == 3
        @test_throws ArgumentError random_credal_set(Random.Xoshiro(3), 4, 0)
    end
end
