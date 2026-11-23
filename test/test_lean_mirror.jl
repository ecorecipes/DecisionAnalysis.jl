# Julia tests that mirror three lemmas of the Lean development (proofs/), stated in
# its notation in the comments. Each is checked on random rational problems with
# oracles written from the definitions.

# The first index attaining the maximum: Lean's `firstArgmax`.
function oracle_first_argmax(f)
    best = 1
    for i in eachindex(f)
        f[i] > f[best] && (best = i)
    end
    return best
end

first_index(result) = findfirst(==(first_choice(result)), result.actions)

@testset "mirrors of Lean lemmas" begin
    @testset "IsMaximal.not_dominated_on_hull" begin
        # theorem IsMaximal.not_dominated_on_hull (h : IsMaximal u P a) :
        #   ¬ ∃ a', ∀ w, (∀ i, 0 ≤ w i) → ∑ i, w i = 1 →
        #     expectedUtility u (mixture w P) a < expectedUtility u (mixture w P) a'
        # A maximal action is not strictly beaten by one rival at every prior of the
        # hull: for each rival there is a mixture at which it does not win.
        rng = Random.Xoshiro(20261205)
        strict = 0
        for _ in 1:80
            m, n, K = rand(rng, 2:5), rand(rng, 2:4), rand(rng, 1:4)
            table = random_payoff_table(rng, m, n; T=BigRat)
            credal = random_credal_set(rng, n, K; T=BigRat)
            points = extreme_points(credal)
            maximal = chosen_indices(decide(DecisionProblem(table, credal), Maximal()))
            mixtures = [[sum(w[k] * points[k][t] for k in 1:K) for t in 1:n]
                        for w in simplex_grid(K, 5)]
            eus = [oracle_expected_utility(table.Q, q) for q in mixtures]
            for a in 1:m, rival in 1:m
                beaten_everywhere = all(eu -> eu[a] < eu[rival], eus)
                if a in maximal
                    @test !beaten_everywhere
                else
                    strict += beaten_everywhere
                end
            end
            # The definition itself (`IsMaximal`): not strictly beaten at every
            # generating point; and an action that is not maximal is beaten by one
            # rival at every prior of the hull, the grid included.
            member = [oracle_expected_utility(table.Q, q) for q in points]
            for a in 1:m
                dominated = any(r -> all(e -> e[a] < e[r], member), 1:m)
                @test (a in maximal) == !dominated
                dominated && @test any(r -> all(eu -> eu[a] < eu[r], eus), 1:m)
            end
        end
        @test strict > 0
    end

    @testset "firstArgmax_congr_of_order" begin
        # theorem firstArgmax_congr_of_order {f : ι → K} {g : ι → K'}
        #   (h : ∀ i j, f i ≤ f j ↔ g i ≤ g j) : firstArgmax g = firstArgmax f
        # `first_choice` depends only on the ranking of the scores, whatever ordered
        # type they live in.
        rng = Random.Xoshiro(20261206)
        ties = 0
        for _ in 1:200
            m = rand(rng, 1:6)
            f = BigRat.(rand(rng, -3:3, m)) .// rand(rng, 1:3)
            # Three rankings equivalent to f, in three ordered types: a strictly
            # increasing rational map, the integer ranks, and floating point.
            cube = [x^3 + 2x - 5 for x in f]
            ranks = [count(<(x), f) for x in f]
            floats = Float64.(ranks) .- 0.5
            for g in (cube, ranks, floats)
                @test all((f[i] <= f[j]) == (g[i] <= g[j]) for i in 1:m, j in 1:m)
                @test oracle_first_argmax(g) == oracle_first_argmax(f)
            end
            ties += count(==(maximum(f)), f) > 1
            # The package: the scores of a known state are the payoffs in it, so an
            # order-equivalent column gives the same first choice and the same ties.
            column(v) = DecisionProblem(PayoffTable(reshape(v, m, 1)), PointBelief(:s1))
            base = decide(column(f), MaxExpectedUtility())
            @test first_index(base) == oracle_first_argmax(f)
            for g in (cube, ranks, floats)
                other = decide(column(g), MaxExpectedUtility(); rtol=0)
                @test first_choice(other) == first_choice(base)
                @test other.chosen == base.chosen
            end
            # Maximin: an increasing map of every payoff commutes with the minimum
            # over states, so the ranking of the scores is unchanged.
            n = rand(rng, 1:4)
            Q = BigRat.(rand(rng, -3:3, m, n))
            scen = ScenarioSet([Symbol(:s, j) for j in 1:n])
            a = decide(DecisionProblem(PayoffTable(Q), scen), Maximin())
            b = decide(DecisionProblem(PayoffTable(Q .^ 3 .+ 2 .* Q), scen), Maximin())
            @test first_choice(a) == first_choice(b) && a.chosen == b.chosen
            @test first_index(a) ==
                  oracle_first_argmax([minimum(Q[i, :]) for i in 1:m])
        end
        @test ties > 20
    end

    @testset "bayesAction_rescale" begin
        # theorem bayesAction_rescale {s : K} (hs : 0 < s) (b : Θ → K) (u p) :
        #   bayesAction (rescale s b u) p = bayesAction u p
        # with rescale s b u a θ = s * u a θ + b θ: the selected Bayes action is
        # unchanged by a positive rescaling and a state-dependent shift.
        rng = Random.Xoshiro(20261207)
        ties = 0
        for _ in 1:200
            m, n = rand(rng, 1:5), rand(rng, 1:4)
            Q = BigRat.(rand(rng, -4:4, m, n))
            prior = random_belief(rng, n; T=BigRat)
            s = rand(rng, 1:9) // rand(rng, 1:9)
            b = BigRat.(rand(rng, -20:20, n)) .// rand(rng, 1:5)
            rescaled = s .* Q .+ permutedims(b)
            before = decide(DecisionProblem(PayoffTable(Q), prior), MaxExpectedUtility())
            after = decide(DecisionProblem(PayoffTable(rescaled), prior),
                           MaxExpectedUtility())
            @test first_choice(after) == first_choice(before)
            @test after.chosen == before.chosen
            eu = oracle_expected_utility(Q, prior.probabilities)
            @test first_index(before) == oracle_first_argmax(eu)
            @test first_index(after) ==
                  oracle_first_argmax(oracle_expected_utility(rescaled,
                                                              prior.probabilities))
            # The expected utilities move by s and a constant (`expectedUtility_rescale`).
            shift = sum(prior.probabilities[t] * b[t] for t in 1:n)
            @test after.scores == s .* before.scores .+ shift
            ties += length(before.chosen) > 1
            # A negative scale reverses the ranking, so the hypothesis 0 < s matters.
            if m > 1 && length(unique(eu)) > 1
                flipped = decide(DecisionProblem(PayoffTable(-s .* Q .+ permutedims(b)),
                                                 prior), MaxExpectedUtility())
                @test flipped.chosen != before.chosen
            end
        end
        @test ties > 5
    end
end
