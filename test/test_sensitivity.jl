# Sensitivity: policy regions as an exact upper envelope, thresholds, sweeps and rank
# reversals, against brute force on a grid.

@testset "sensitivity" begin
    @testset "policy regions against a grid" begin
        rng = Xoshiro(31)
        for _ in 1:60
            m, n = rand(rng, 2:5), rand(rng, 2:4)
            base = exact_problem(rng, m, n)
            direction = random_payoff_table(rng, m, n; T=BigRat)
            from, to = BigRat(-1), BigRat(2)
            regions = policy_regions(base, direction; from, to)
            p = base.belief.probabilities
            α = oracle_expected_utility(base.payoff.Q, p)
            β = oracle_expected_utility(direction.Q, p)
            # The regions tile the interval.
            @test first(regions).from == from && last(regions).to == to
            @test all(regions[k].to == regions[k + 1].from for k in 1:(length(regions) - 1))
            @test all(r -> r.from < r.to, regions)
            @test all(regions[k].actions != regions[k + 1].actions
                      for k in 1:(length(regions) - 1))
            actions = action_labels(base)
            # Inside a region, the optimal set is the region's, by brute force.
            for r in regions
                inner = [r.from + (r.to - r.from) * q for q in (1 // 7, 1 // 2, 6 // 7)]
                for best in oracle_best_on_grid(α, β, inner)
                    @test actions[best] == r.actions
                end
            end
            # Every point of a fine grid is covered by the region it falls in.
            for t in from:(1 // 16):to
                best = actions[only(oracle_best_on_grid(α, β, [t]))]
                @test any(r -> r.from <= t <= r.to && issubset(r.actions, best), regions)
            end
            thresholds = one_way_thresholds(base, direction; from, to)
            @test length(thresholds) == length(regions) - 1
            for th in thresholds
                v = α .+ th.threshold .* β
                # At a threshold the actions of both sides tie for best.
                @test all(a -> v[findfirst(==(a), actions)] == maximum(v),
                          vcat(th.below, th.above))
                @test th.threshold isa Rational
            end
        end
    end

    @testset "the infection weight" begin
        uninfected = PayoffTable([:none, :treat, :cull], [:low, :high],
                                 [90 40; 80 70; 50 60])
        infected = PayoffTable([:none, :treat, :cull], [:low, :high],
                               [10 50; 30 40; 0 2])
        prior = [1 // 2, 1 // 2]
        base = DecisionProblem(uninfected, prior; baseline=:none)
        regions = policy_regions(base, -infected; from=0, to=3)
        # E[u] = 65, 75, 55; E[i] = 30, 35, 1: treat until 75 − 35w = 55 − w.
        @test regions == [(; from=0 // 1, to=10 // 17, actions=[:treat]),
                          (; from=10 // 17, to=3 // 1, actions=[:cull])]
        @test one_way_thresholds(base, -infected; from=0, to=3) ==
              [(; threshold=10 // 17, below=[:treat], above=[:cull])]
        @test isempty(one_way_thresholds(base, -infected; from=1, to=3))

        build(w) = DecisionProblem(infection_payoff(; uninfected, infected, weight=w),
                                   prior; baseline=:none)
        sweep = weight_sweep(build, 0:(1 // 4):3)
        @test Tables.istable(sweep)
        for row in sweep
            region = only(filter(r -> r.from <= row.weight < r.to ||
                                      (row.weight == 3 && r.to == 3), regions))
            @test [row.first_choice] == region.actions
            @test row.ranking[1] == row.first_choice
            @test row.scores == aggregate(Expectation(), build(row.weight))
        end
        reversals = rank_reversals(sweep)
        @test any(r -> r.top_changed && r.from == 1 // 2 && r.to == 3 // 4 &&
                       Set(r.actions) == Set([:treat, :cull]), reversals)
        # none and cull swap when 65 − 30w = 55 − w, at w = 10/29, without changing the
        # first choice.
        @test any(r -> !r.top_changed && Set(r.actions) == Set([:none, :cull]) &&
                       r.from <= 10 // 29 <= r.to, reversals)
        # A sweep under another rule.
        maximin = weight_sweep(build, [0, 1]; rule=Maximin())
        @test maximin[1].first_choice == decide(build(0), Maximin()).chosen[1]

        # Two problems side by side.
        @test rank_reversals(build(0), build(0)) == []
        r = rank_reversals(build(0), build(3))
        @test (; actions=(:treat, :cull), top_changed=true) in r
        @test all(x -> x.top_changed, r)
        @test rank_reversals(build(0), build(3); aggregator=WorstCase()) isa Vector

        @test_throws ArgumentError policy_regions(base, -infected; from=1, to=1)
        @test_throws LabelMismatchError policy_regions(base, PayoffTable([1 2; 3 4; 5 6]))
        @test_throws BeliefFormError policy_regions(DecisionProblem(uninfected,
                                                                    ScenarioSet([:low])),
                                                    -infected)
        @test_throws ArgumentError weight_sweep(w -> 1, [0, 1])
        @test_throws LabelMismatchError rank_reversals(build(0),
                                                       DecisionProblem(uninfected, prior;
                                                                       feasible=[:none,
                                                                                 :treat]))
        # Floating point and a sample belief go through the same code.
        float_regions = policy_regions(DecisionProblem(PayoffTable(Float64.(uninfected.Q);
                                                                   actions=uninfected.actions),
                                                       SampleBelief(2)),
                                       PayoffTable(-Float64.(infected.Q);
                                                   actions=infected.actions); from=0, to=3)
        @test length(float_regions) == 2 && float_regions[1].to ≈ 10 / 17
    end
end
