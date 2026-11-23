# Rules over credal sets. E-admissibility is checked three ways: an exact oracle for
# two generators, the self-certifying witness, and a grid search for the complement.

# Exact oracle for a credal set with two generators. The advantage of action `a` over
# `b` along the segment is linear in the mixing weight, so the minimum over `b` is
# concave and piecewise linear: its maximum is at an end point or where two of the
# lines cross.
function oracle_e_admissible_segment(E)
    m = size(E, 1)
    out = Int[]
    for a in 1:m
        D = [(E[a, 1] - E[b, 1], E[a, 2] - E[b, 2]) for b in 1:m]
        value(λ) = minimum((1 - λ) * d[1] + λ * d[2] for d in D)
        candidates = BigRat[0, 1]
        for i in 1:m, j in (i + 1):m
            # (1 − λ) d1 + λ d2 equal for i and j
            denom = (D[i][2] - D[i][1]) - (D[j][2] - D[j][1])
            iszero(denom) && continue
            λ = (D[j][1] - D[i][1]) / denom
            0 <= λ <= 1 && push!(candidates, λ)
        end
        any(λ -> value(λ) >= 0, candidates) && push!(out, a)
    end
    return out
end

# All weight vectors on the simplex with entries k/steps.
function compositions(K, total)
    K == 1 && return [[total]]
    return [vcat(k, rest) for k in 0:total for rest in compositions(K - 1, total - k)]
end
simplex_grid(K, steps) = [c .// steps for c in compositions(K, steps)]

@testset "credal sets" begin
    Q = [0 -10 -60 -150; -20 -20 -25 -90; -48 -48 -48 -48]
    t = PayoffTable([:nothing, :levee, :relocate], [:low, :moderate, :high, :extreme], Q)
    ensemble = CredalSet(t.states,
                         [[1 // 2, 3 // 10, 3 // 20, 1 // 20],
                          [7 // 20, 3 // 10, 1 // 4, 1 // 10],
                          [3 // 5, 1 // 4, 3 // 25, 3 // 100]])
    flood = DecisionProblem(t, ensemble)

    @testset "flood ensemble" begin
        r = decide(flood, GammaMaximin())
        @test r.scores == [-33, -113 // 4, -48]
        @test r.chosen == [:levee]
        @test decide(flood, EAdmissible()).chosen == [:nothing, :levee]
        @test decide(flood, Maximal()).chosen == [:nothing, :levee]
        id = decide(flood, IntervalDominance())
        @test id.chosen == [:nothing, :levee]
        @test id.details.lower == [-33, -113 // 4, -48]
        @test id.details.upper == aggregate(UpperExpectation(), flood)
        @test decide(flood, DecisionRule(UpperExpectation())).chosen == [:nothing]
        @test decide(flood, DecisionRule(AlphaMaxmin(1))).chosen == [:levee]
        @test decide(flood, DecisionRule(AlphaMaxmin(0))).chosen == [:nothing]
    end

    @testset "E-admissibility where enumerating extreme points is wrong" begin
        # :hedge is best at no generator of the set, but is best at the midpoint.
        payoff = PayoffTable([:left, :right, :hedge], [:s1, :s2],
                             [1//1 0; 0 1; 3//5 3//5])
        credal = CredalSet([:s1, :s2], [[9 // 10, 1 // 10], [1 // 10, 9 // 10]])
        problem = DecisionProblem(payoff, credal)
        by_enumeration = unique(reduce(vcat,
                                       [decide(DecisionProblem(payoff, q),
                                               MaxExpectedUtility()).chosen
                                        for q in extreme_points(credal)]))
        @test sort(by_enumeration) == [:left, :right]
        r = decide(problem, EAdmissible())
        @test r.chosen == [:left, :right, :hedge]
        # The witness for :hedge is an interior prior at which it is a Bayes action.
        w = r.details.witnesses[:hedge]
        @test sum(w) == 1 && all(>(0), w) && w ∉ extreme_points(credal)
        @test :hedge in decide(DecisionProblem(payoff, w), MaxExpectedUtility()).chosen
        # The same in floating point.
        fproblem = DecisionProblem(PayoffTable([:left, :right, :hedge], [:s1, :s2],
                                               [1.0 0.0; 0.0 1.0; 0.6 0.6]),
                                   CredalSet([:s1, :s2], [[0.9, 0.1], [0.1, 0.9]]))
        @test decide(fproblem, EAdmissible()).chosen == [:left, :right, :hedge]
        # A hedge that is never best: maximal (nothing beats it everywhere) but not
        # E-admissible. The inclusion E-admissible ⊆ maximal is strict here.
        weak = PayoffTable([:left, :right, :hedge], [:s1, :s2],
                           [1//1 0; 0 1; 2//5 2//5])
        vacuous = CredalSet([:s1, :s2], [[1 // 1, 0 // 1], [0 // 1, 1 // 1]])
        wp = DecisionProblem(weak, vacuous)
        @test decide(wp, EAdmissible()).chosen == [:left, :right]
        @test decide(wp, Maximal()).chosen == [:left, :right, :hedge]
        # Interval dominance is weaker still: :bad is beaten by :good at every prior, so
        # it is not maximal, but the two intervals of expected utility overlap.
        dominated = PayoffTable([:good, :bad], [:s1, :s2], [2 3; 1 5//2])
        dd = DecisionProblem(dominated, vacuous)
        @test decide(dd, Maximal()).chosen == [:good]
        @test decide(dd, IntervalDominance()).chosen == [:good, :bad]
    end

    @testset "properties on random problems" begin
        rng = Random.Xoshiro(2026)
        for trial in 1:60
            m, n = rand(rng, 2:5), rand(rng, 2:4)
            K = trial <= 30 ? 2 : rand(rng, 1:4)
            payoff = random_payoff_table(rng, m, n; T=BigRat)
            credal = random_credal_set(rng, n, K; T=BigRat)
            problem = DecisionProblem(payoff, credal)
            Q, pts = payoff.Q, extreme_points(credal)
            E = reduce(hcat, [oracle_expected_utility(Q, q) for q in pts])

            gm = decide(problem, GammaMaximin())
            ea = decide(problem, EAdmissible())
            mx = decide(problem, Maximal())
            id = decide(problem, IntervalDominance())
            e_set, m_set, i_set = chosen_indices(ea), chosen_indices(mx), chosen_indices(id)

            # Γ-maximin: the value is below every member's expected utility of the
            # chosen action, and so below every member's best.
            value = maximum(gm.scores)
            for a in chosen_indices(gm), k in 1:K
                @test value <= E[a, k] <= maximum(E[:, k])
            end
            @test gm.scores == [minimum(E[a, :]) for a in 1:m]

            # Inclusions: Bayes actions of members ⊆ E-admissible ⊆ maximal ⊆ interval
            # dominance, and Γ-maximin choices are maximal.
            for k in 1:K
                @test issubset(argmax_set(E[:, k]), e_set)
            end
            @test issubset(e_set, m_set)
            @test issubset(m_set, i_set)
            @test issubset(chosen_indices(gm), m_set)
            @test !isempty(e_set)

            # Maximality and interval dominance against their definitions.
            @test m_set ==
                  [a for a in 1:m if !any(b -> all(E[b, :] .> E[a, :]), 1:m)]
            lower = [minimum(E[a, :]) for a in 1:m]
            @test i_set == [a for a in 1:m if maximum(E[a, :]) >= maximum(lower)]

            # E-admissibility: every witness is a prior in the set under which the
            # action is a Bayes action...
            for a in e_set
                w = ea.details.witnesses[payoff.actions[a]]
                @test sum(w) == 1 && all(>=(0), w)
                @test a in argmax_set(oracle_expected_utility(Q, w))
            end
            # ...the Bayes action at a random interior prior is E-admissible...
            λ = random_stochastic(rng, BigRat, K, 1)
            interior = sum(λ[k] * pts[k] for k in 1:K)
            @test issubset(argmax_set(oracle_expected_utility(Q, interior)), e_set)
            # ...with two generators the exact oracle agrees...
            K == 2 && @test e_set == oracle_e_admissible_segment(E)
            # ...and no prior on a grid over the set makes a rejected action optimal.
            for w in simplex_grid(K, 12)
                mix = sum(w[k] * E[:, k] for k in 1:K)
                @test issubset(argmax_set(mix), e_set)
            end

            # Floating point agrees with exact arithmetic when there are no near-ties.
            gaps = [abs(E[a, k] - E[b, k]) for a in 1:m, b in 1:m, k in 1:K if a != b]
            if all(>(1 // 1000), gaps)
                fproblem = DecisionProblem(PayoffTable(Float64.(Q)),
                                           CredalSet([Float64.(q) for q in pts]))
                @test chosen_indices(decide(fproblem, Maximal())) == m_set
                @test chosen_indices(decide(fproblem, IntervalDominance())) == i_set
                @test chosen_indices(decide(fproblem, GammaMaximin())) ==
                      chosen_indices(gm)
            end
        end
    end

    @testset "a single prior gives the Bayes actions" begin
        rng = Random.Xoshiro(8)
        for _ in 1:20
            problem = exact_problem(rng, rand(rng, 1:5), rand(rng, 1:4))
            bayes = decide(problem, MaxExpectedUtility()).chosen
            for rule in (EAdmissible(), Maximal(), IntervalDominance(), GammaMaximin())
                @test decide(problem, rule).chosen == bayes
            end
        end
    end
end
