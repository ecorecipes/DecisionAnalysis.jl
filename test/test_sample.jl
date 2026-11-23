# Sample beliefs: construction, and the Phase 1 machinery acting on a sample exactly as
# it does on the finite belief with the same probabilities.

@testset "sample beliefs" begin
    @testset "construction and weights" begin
        b = SampleBelief((; r=[0.1, 0.2, 0.3], k=[1, 2, 3]))
        @test state_labels(b) == [:s1, :s2, :s3]
        @test probabilities(b) == fill(1 // 3, 3)
        @test probabilities(b) isa Vector{Rational{Int}}
        @test draw_weights(b) === probabilities(b)
        @test parameter_draws(b) == (; r=[0.1, 0.2, 0.3], k=[1, 2, 3])
        @test length(b) == 3
        @test effective_sample_size(b) == 3

        # A vector of named tuples is a table too; weights and labels from columns.
        rows = [(; id="a", r=0.1, w=1), (; id="b", r=0.2, w=3)]
        c = SampleBelief(rows; weights=:w, labels=:id)
        @test state_labels(c) == [:a, :b]
        @test probabilities(c) == [1 // 4, 3 // 4]
        @test keys(parameter_draws(c)) == (:r,)
        @test effective_sample_size(c) == 1 // (1 // 16 + 9 // 16)

        # Floating-point weights are normalised in floating point.
        d = SampleBelief((; r=[1.0, 2.0]); weights=[0.5, 1.5])
        @test probabilities(d) ≈ [0.25, 0.75]
        @test probabilities(d) isa Vector{Float64}

        e = SampleBelief(4; weights=[1, 1, 1, 1])
        @test isempty(parameter_draws(e)) && length(e) == 4
        @test FiniteBelief(c) == FiniteBelief([:a, :b], [1 // 4, 3 // 4])
        @test c == SampleBelief(rows; weights=:w, labels=:id)
        @test hash(c) == hash(SampleBelief(rows; weights=:w, labels=:id))
        @test occursin("2 draws", sprint(show, c))

        @test_throws InvalidBeliefError SampleBelief((; r=[1.0, 2.0]); weights=[1, -1])
        @test_throws InvalidBeliefError SampleBelief((; r=[1.0, 2.0]); weights=[0, 0])
        @test_throws InvalidBeliefError SampleBelief((; r=[1.0, 2.0]); weights=[1.0, NaN])
        @test_throws InvalidBeliefError SampleBelief((; r=[1.0, 2.0]); weights=[1, 2, 3])
        @test_throws InvalidBeliefError SampleBelief((; r=[1.0, 2.0]); labels=[:a, :a])
        @test_throws InvalidBeliefError SampleBelief(0)
        @test_throws ArgumentError SampleBelief((; r=[1.0, 2.0]); weights=:nope)
        @test_throws ArgumentError SampleBelief((;))
        @test_throws ArgumentError SampleBelief(3.5)
    end

    @testset "rules on a sample agree with the finite belief of the same weights" begin
        rng = Xoshiro(20261123)
        for _ in 1:40
            m, n = rand(rng, 2:5), rand(rng, 2:7)
            t = random_payoff_table(rng, m, n; T=BigRat)
            w = rand(rng, 1:6, n)
            sample = DecisionProblem(t, SampleBelief(n; weights=w))
            finite = DecisionProblem(t, FiniteBelief(t.states, BigRat.(w) .// sum(w)))
            p = BigRat.(w) .// sum(w)
            Q = t.Q
            # Slow oracles, not the package's own finite path.
            @test aggregate(Expectation(), sample) == oracle_expected_utility(Q, p)
            @test evpi(sample) == oracle_evpi(Q, p)
            @test regret_table(sample) == oracle_regret(Q)
            @test expected_regret(sample) == oracle_expected_utility(oracle_regret(Q), p)
            @test minimum(expected_regret(sample)) == evpi(sample)
            @test evpi(sample) isa Rational
            for rule in (MaxExpectedUtility(), Maximin(), Maximax(), MinimaxRegret(),
                         DecisionRule(CVaR(1 // 3)), DecisionRule(Quantile(1 // 2)),
                         DecisionRule(Hurwicz(1 // 4)))
                a, b = decide(sample, rule), decide(finite, rule)
                @test a.scores == b.scores
                @test a.chosen == b.chosen
            end
            partition = rand(rng, 1:2, n)
            L = partition_likelihood(partition, BigRat)
            @test evppi(sample, partition) == oracle_evsi(Q, p, L)
            e = random_stochastic(rng, BigRat, 3, n)
            @test evsi(sample, Experiment(e)) == oracle_evsi(Q, p, e)
        end
    end

    @testset "what is flagged and what is refused" begin
        t = PayoffTable([:none, :act], [:s1, :s2, :s3], [3 1 2; 2 2 2])
        p = DecisionProblem(t, SampleBelief(3); baseline=:none)
        for rule in (Maximin(), Maximax(), DecisionRule(Hurwicz(1 // 2)))
            r = decide(p, rule)
            @test r.details.sample_dependent
            @test occursin("sample", r.details.sample_note)
            @test occursin("sample", sprint(show, MIME"text/plain"(), r))
        end
        r = decide(p, MinimaxRegret())
        @test r.details.sample_dependent && r.details.menu_dependent
        @test !haskey(decide(p, MaxExpectedUtility()).details, :sample_dependent)
        @test !haskey(decide(DecisionProblem(t, [1 // 3, 1 // 3, 1 // 3]),
                             Maximin()).details,
                      :sample_dependent)

        @test_throws BeliefFormError aggregate(Laplace(), p)
        @test_throws BeliefFormError total_variation(SampleBelief(3), SampleBelief(3))
        @test_throws BeliefFormError bayes_invert(SampleBelief(3),
                                                  Experiment([1//2 1//2 1//2;
                                                              1//2 1//2 1//2]))
        @test_throws BeliefFormError export_decision_certificate(IOBuffer(), p)
        @test_throws BeliefFormError export_eadmissible_witness(IOBuffer(), p, :act)
        # The way out that the message names works.
        finite = DecisionProblem(t, FiniteBelief(SampleBelief(3)))
        io = IOBuffer()
        export_decision_certificate(io, finite)
        @test occursin("decision-certificate/1", String(take!(io)))

        @test expected_gain_over_baseline(p) == [0, 0]
        @test_throws LabelMismatchError DecisionProblem(t, SampleBelief(2))
        q = restrict_scenarios(DecisionProblem(t,
                                               SampleBelief((; x=[1, 2, 3]);
                                                            weights=[1, 1, 2])),
                               [:s1, :s3])
        @test probabilities(q.belief) == [1 // 3, 2 // 3]
        @test parameter_draws(q.belief).x == [1, 3]
    end
end
