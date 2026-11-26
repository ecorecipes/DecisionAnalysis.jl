# Risk attitude and multi-attribute utilities. The oracles are the textbook formulas
# written out with explicit loops over the raw rows of runs; they share no code with
# the package.

# A table of runs with counts of uninfected and infected frogs out of 60, unequal
# numbers of replicates and unequal draw weights.
function utility_runs(rng; draws=4, actions=(:none, :treat, :cull), replicates=5)
    rows = NamedTuple[]
    for d in 1:draws, a in actions, r in 1:replicates
        # Drop a few replicates so that cells differ in size (never the first two).
        r > 2 && rand(rng) < 0.2 && continue
        uninfected = rand(rng, 1:40)
        infected = rand(rng, 0:(60 - uninfected))
        push!(rows, (; draw=d, action=a, replicate=r, uninfected, infected))
    end
    return rows
end

# Mean utility of the runs of one cell, one run at a time.
function oracle_cell_utilities(rows, d, a, f)
    us = Float64[]
    for row in rows
        row.draw == d && row.action == a || continue
        push!(us, f(row))
    end
    return us
end

@testset "risk attitude and multi-attribute utilities" begin
    @testset "single-attribute utilities: closed forms" begin
        rng = Xoshiro(41)
        for _ in 1:20
            x = rand(rng, 1:60, rand(rng, 2:6))
            w = rand(rng, 1:4, length(x))
            p = w ./ sum(w)
            mean_x = sum(p .* x)

            # Linear: the certainty equivalent is the mean, exactly, and the premium zero.
            lin = LinearUtility(:n)
            @test certainty_equivalent(lin, x; weights=w) == sum(w .* x) // sum(w)
            @test iszero(risk_premium(lin, x; weights=w))
            @test certainty_equivalent(IndividualsUtility(:n), x; weights=w) ==
                  sum(w .* x) // sum(w)

            # Exponential (CARA): CE = −log(E exp(−r X)) / r.
            r = rand(rng, (0.02, 0.1, -0.03))
            cara = ExponentialUtility(:n; risk_aversion=r)
            ce = -log(sum(p[i] * exp(-r * x[i]) for i in eachindex(x))) / r
            @test certainty_equivalent(cara, x; weights=w) ≈ ce
            @test risk_premium(cara, x; weights=w) ≈ mean_x - ce
            @test sign(risk_premium(cara, x; weights=w)) in (0, sign(r))
            # Constant absolute risk aversion: a shift of the lottery shifts the CE.
            @test certainty_equivalent(cara, x .+ 7; weights=w) ≈ ce + 7
            @test inverse_utility(cara, cara(x[1])) ≈ x[1]

            # Power (CRRA): CE = (E X^(1−ρ))^(1/(1−ρ)), the geometric mean at ρ = 1.
            for ρ in (0.5, 2.0, -0.5)
                crra = PowerUtility(:n; risk_aversion=ρ)
                ce = sum(p[i] * Float64(x[i])^(1 - ρ) for i in eachindex(x))^(1 / (1 - ρ))
                @test certainty_equivalent(crra, x; weights=w) ≈ ce
                # Constant relative risk aversion: scaling the lottery scales the CE.
                @test certainty_equivalent(crra, 3 .* x; weights=w) ≈ 3 * ce
                @test sign(risk_premium(crra, x; weights=w)) in (0, sign(ρ))
            end
            geometric = exp(sum(p[i] * log(x[i]) for i in eachindex(x)))
            @test certainty_equivalent(PowerUtility(:n; risk_aversion=1), x; weights=w) ≈
                  geometric
        end
        # Risk aversion zero is risk neutrality in both families.
        @test certainty_equivalent(ExponentialUtility(:n; risk_aversion=0), [0, 40]) == 20
        @test certainty_equivalent(PowerUtility(:n; risk_aversion=0), [0, 40]) ≈ 20
        # A small risk aversion is close to it.
        @test certainty_equivalent(ExponentialUtility(:n; risk_aversion=1e-9), [0, 40]) ≈
              20 atol = 1e-5
        # Callable on a number and on the named outcomes of a run.
        u = ExponentialUtility(:frogs; risk_aversion=0.05)
        @test u(12) == u((; frogs=12, eggs=3)) ≈ (1 - exp(-0.6)) / 0.05
        @test_throws InvalidOutcomeError u((; eggs=3))
    end

    @testset "scaling to [0, 1]" begin
        for u in (LinearUtility(:n; worst=5, best=45),
                  ExponentialUtility(:n; risk_aversion=0.08, worst=5, best=45),
                  ExponentialUtility(:n; risk_aversion=-0.04, worst=5, best=45),
                  PowerUtility(:n; risk_aversion=0.5, worst=5, best=45),
                  PowerUtility(:n; risk_aversion=1, worst=5, best=45),
                  PowerUtility(:n; risk_aversion=3, worst=5, best=45),
                  # Outcomes that count against: best below worst.
                  LinearUtility(:n; worst=45, best=5),
                  ExponentialUtility(:n; risk_aversion=0.08, worst=45, best=5))
            @test u(u.worst) ≈ 0 atol = 1e-12
            @test u(u.best) ≈ 1
            xs = 5:8:45
            values = [u(x) for x in xs]
            @test issorted(values; rev=u.best < u.worst)
            @test all(inverse_utility(u, u(x)) ≈ x for x in xs)
            # A positive affine map of a utility has the same certainty equivalents.
            natural = u isa LinearUtility ? LinearUtility(:n) :
                      u isa PowerUtility ?
                      PowerUtility(:n; risk_aversion=u.risk_aversion) : nothing
            natural === nothing || u.best < u.worst ||
                @test certainty_equivalent(u, [10, 20, 40]) ≈
                      certainty_equivalent(natural, [10, 20, 40])
        end
        @test LinearUtility(:n; worst=0, best=50)(20) === 2 // 5      # exact
        @test LinearUtility(:n; worst=50, best=0)(20) === 3 // 5
        # Risk aversion over the improvement: for an outcome that counts against, the
        # certainty equivalent is worse (larger) than the mean and the premium positive.
        against = ExponentialUtility(:infected; risk_aversion=0.1, worst=50, best=0)
        @test certainty_equivalent(against, [0, 50]) > 25
        @test risk_premium(against, [0, 50]) ≈ certainty_equivalent(against, [0, 50]) - 25
        @test risk_premium(against, [0, 50]) > 0

        @test_throws ArgumentError LinearUtility(:n; worst=1)
        @test_throws ArgumentError LinearUtility(:n; worst=1, best=1)
        @test_throws ArgumentError ExponentialUtility(:n; risk_aversion=Inf)
        @test_throws ArgumentError PowerUtility(:n; risk_aversion=0.5, worst=10, best=2)
        @test_throws ArgumentError PowerUtility(:n; risk_aversion=1, worst=0, best=2)
        @test_throws InvalidOutcomeError PowerUtility(:n; risk_aversion=0.5)(-1)
        @test_throws InvalidOutcomeError PowerUtility(:n; risk_aversion=2)(0)
        @test_throws ArgumentError inverse_utility(ExponentialUtility(:n; risk_aversion=1),
                                                   2)
        @test_throws ArgumentError certainty_equivalent(LinearUtility(:n), Int[])
        @test_throws ArgumentError certainty_equivalent(LinearUtility(:n), [1, 2];
                                                        weights=[1])
    end

    @testset "a non-linear utility is applied per run, before averaging" begin
        rng = Xoshiro(42)
        for _ in 1:10
            rows = utility_runs(rng)
            t = OutcomeTable(rows; outcomes=[:uninfected, :infected])
            r = rand(rng, (0.03, 0.1))
            for (u, f) in ((ExponentialUtility(:uninfected; risk_aversion=r),
                            row -> (1 - exp(-r * row.uninfected)) / r),
                           (PowerUtility(:uninfected; risk_aversion=0.5),
                            row -> sqrt(row.uninfected) / 0.5),
                           (PowerUtility(:uninfected; risk_aversion=1),
                            row -> log(row.uninfected)))
                table = payoff_table(t, u)
                for (d, draw) in enumerate(1:4),
                    (a, action) in enumerate((:none, :treat, :cull))
                    us = oracle_cell_utilities(rows, draw, action, f)
                    m, se = oracle_cell(us)
                    @test table.Q[a, d] ≈ m
                    @test table.standard_errors[a, d] ≈ se
                    @test table.replicates[a, d] == length(us)
                    # Jensen: the utility of the mean count is larger than the mean
                    # utility for a concave utility, so averaging first would be wrong.
                    xs = oracle_cell_utilities(rows, draw, action, row -> row.uninfected)
                    @test u(sum(xs) / length(xs)) >= table.Q[a, d] - 1e-12
                end
                # The two orders differ somewhere, so the test above is not vacuous.
                wrong = [u(sum(oracle_cell_utilities(rows, d, a, row -> row.uninfected)) /
                           length(oracle_cell_utilities(rows, d, a, row -> row.uninfected)))
                         for a in (:none, :treat, :cull), d in 1:4]
                @test maximum(abs.(wrong .- table.Q)) > 1e-6

                # Gains over the baseline are differences of utilities, run by run.
                gains = payoff_table(t, u; baseline=:none)
                @test all(iszero, gains.Q[1, :])
                paired = Float64[]
                for row in rows
                    row.draw == 2 && row.action == :treat || continue
                    # Only the replicates that the baseline also has are paired.
                    base = filter(x -> x.draw == 2 && x.action == :none &&
                                       x.replicate == row.replicate, rows)
                    isempty(base) || push!(paired, f(row) - f(only(base)))
                end
                @test gains.Q[2, 2] ≈ sum(paired) / length(paired)
            end

            # The certainty equivalent of each action, against explicit loops: a draw
            # with its weight, then a replicate of the cell with equal chances.
            weights = [3, 1, 2, 2]
            tw = OutcomeTable(rows; outcomes=[:uninfected, :infected], weights)
            u = ExponentialUtility(:uninfected; risk_aversion=r)
            summary = certainty_equivalent(tw, u)
            problem = DecisionProblem(tw, u)
            scores = aggregate(Expectation(), problem)
            for (a, action) in enumerate((:none, :treat, :cull))
                eu, ex = 0.0, 0.0
                for d in 1:4
                    us = oracle_cell_utilities(rows, d, action,
                                               row -> (1 - exp(-r * row.uninfected)) / r)
                    xs = oracle_cell_utilities(rows, d, action, row -> row.uninfected)
                    eu += weights[d] / 8 * sum(us) / length(us)
                    ex += weights[d] / 8 * sum(xs) / length(xs)
                end
                ce = -log(1 - r * eu) / r
                @test summary[a].action === action
                @test summary[a].expected_utility ≈ eu ≈ scores[a]
                @test summary[a].certainty_equivalent ≈ ce
                @test summary[a].expected_outcome ≈ ex
                @test summary[a].risk_premium ≈ ex - ce
                @test summary[a].risk_premium > 0
                @test inverse_utility(u, scores[a]) ≈ ce
            end
            # A linear utility returns the expected count, exactly.
            linear = certainty_equivalent(tw, LinearUtility(:uninfected))
            @test all(row.certainty_equivalent == row.expected_outcome &&
                          row.certainty_equivalent isa Rational for row in linear)
            @test [row.certainty_equivalent for row in linear] ==
                  aggregate(Expectation(),
                            DecisionProblem(tw, IndividualsUtility(:uninfected)))
            @test_throws ArgumentError certainty_equivalent(tw, InfectionUtility())
        end
    end

    @testset "InfectionUtility is an additive utility" begin
        rng = Xoshiro(43)
        for _ in 1:10
            rows = utility_runs(rng)
            t = OutcomeTable(rows; outcomes=[:uninfected, :infected])
            w = rand(rng, (0, 1 // 2, 1, 3))
            reference = payoff_table(t, InfectionUtility(; weight=w))
            additive = payoff_table(t, AdditiveUtility(:uninfected => 1, :infected => -w))
            @test additive.Q == reference.Q && additive.Q isa Matrix{<:Rational}
            @test additive.standard_errors ≈ reference.standard_errors
            @test additive.replicates == reference.replicates
            @test payoff_table(t,
                               AdditiveUtility([LinearUtility(:uninfected),
                                                LinearUtility(:infected)];
                                               weights=[1, -w])).Q == reference.Q

            # The same utility with both attributes scaled to [0, 1] out of N = 60 and
            # weights summing to one is a positive affine map of it,
            # U = (uninfected − w infected) / (N (1 + w)) + w / (1 + w): the same choice
            # and the same value of information up to the factor.
            N = 60
            scaled = AdditiveUtility(LinearUtility(:uninfected; worst=0, best=N) =>
                                         1 //
                                         (1 + w),
                                     LinearUtility(:infected; worst=N, best=0) =>
                                         w //
                                         (1 + w))
            Qs = payoff_table(t, scaled).Q
            @test Qs == reference.Q .// (N * (1 + w)) .+ w // (1 + w)
            @test all(0 .<= Qs .<= 1)
            p1 = DecisionProblem(t, InfectionUtility(; weight=w))
            p2 = DecisionProblem(t, scaled)
            @test decide(p1, MaxExpectedUtility()).chosen ==
                  decide(p2, MaxExpectedUtility()).chosen
            @test evpi(p2) == evpi(p1) // (N * (1 + w))
        end
        # Components may be any utility of one run, or a plain function.
        mixed = AdditiveUtility(ExponentialUtility(:uninfected; risk_aversion=0.1) => 2,
                                (o -> o.infected^2) => -0.5, InfectionUtility() => 1)
        o = (; uninfected=10, infected=4)
        @test mixed(o) ≈ 2 * (1 - exp(-1)) / 0.1 - 0.5 * 16 + 6
        @test_throws ArgumentError AdditiveUtility()
        @test_throws ArgumentError AdditiveUtility(:a => Inf)
        @test_throws ArgumentError AdditiveUtility([:a, :b]; weights=[1])
    end

    @testset "multiplicative utility (Keeney and Raiffa)" begin
        rng = Xoshiro(44)
        attributes = (LinearUtility(:uninfected; worst=0, best=60),
                      ExponentialUtility(:infected; risk_aversion=0.05, worst=60, best=0),
                      PowerUtility(:tadpoles; risk_aversion=0.5, worst=0, best=400))
        best = (; uninfected=60, infected=0, tadpoles=400)
        worst = (; uninfected=0, infected=60, tadpoles=0)
        for _ in 1:20
            k = [rand(rng, 1:9) // 10 for _ in 1:3]
            u = MultiplicativeUtility(attributes; weights=k)
            K = u.interaction
            if sum(k) == 1
                @test K == 0
            else
                # K is the nonzero root of 1 + K = Π (1 + K k_i), with the right sign.
                @test 1 + K ≈ prod(1 + K * ki for ki in k) rtol = 1e-10
                @test abs(K) > 1e-6
                @test sum(k) > 1 ? -1 < K < 0 : K > 0
            end
            # Corners: everything at its worst is 0, at its best 1, and one attribute at
            # its best with the others at their worst is its weight.
            @test u(worst) ≈ 0 atol = 1e-12
            @test u(best) ≈ 1
            for (i, name) in enumerate((:uninfected, :infected, :tadpoles))
                corner = merge(worst, NamedTuple{(name,)}((best[name],)))
                @test u(corner) ≈ k[i]
            end
            # The definition, from the attribute utilities one by one.
            o = (; uninfected=rand(rng, 0:60), infected=rand(rng, 0:60),
                 tadpoles=rand(rng, 0:400))
            parts = [attributes[1](o.uninfected), attributes[2](o.infected),
                     attributes[3](o.tadpoles)]
            expected = sum(k) == 1 ? sum(k .* parts) :
                       (prod(1 .+ K .* k .* parts) - 1) / K
            @test u(o) ≈ expected
            @test -1e-12 <= u(o) <= 1 + 1e-12
        end
        # Weights that sum to one: exactly the additive form.
        k = (1 // 2, 3 // 10, 1 // 5)
        o = (; uninfected=31, infected=17, tadpoles=100)
        @test MultiplicativeUtility(attributes; weights=k)(o) ==
              AdditiveUtility(attributes; weights=k)(o)
        linear = (LinearUtility(:uninfected; worst=0, best=60),
                  LinearUtility(:infected; worst=60, best=0))
        @test MultiplicativeUtility(linear; weights=(1 // 4, 3 // 4))(o) isa Rational
        # Substitutes (Σk > 1) value a mixed result above the additive form's
        # normalised value; complements (Σk < 1) below.
        half = (; uninfected=60, infected=60)
        @test MultiplicativeUtility(linear; weights=(0.8, 0.8))(half) == 0.8
        @test MultiplicativeUtility(linear; weights=(0.2, 0.2))(half) == 0.2

        # In a payoff table: applied to each run, against explicit loops.
        rows = utility_runs(rng)
        t = OutcomeTable(rows; outcomes=[:uninfected, :infected])
        u = MultiplicativeUtility(linear; weights=(0.7, 0.6))
        K = u.interaction
        table = payoff_table(t, u)
        f(row) = ((1 + K * 0.7 * row.uninfected / 60) *
                  (1 + K * 0.6 * (60 - row.infected) / 60) - 1) / K
        for d in 1:4, (a, action) in enumerate((:none, :treat, :cull))
            m, se = oracle_cell(oracle_cell_utilities(rows, d, action, f))
            @test table.Q[a, d] ≈ m
            @test table.standard_errors[a, d] ≈ se
        end
        @test_throws InvalidOutcomeError u((; uninfected=70, infected=0))
        @test_throws ArgumentError MultiplicativeUtility(linear[1] => 0.5)
        @test_throws ArgumentError MultiplicativeUtility(linear; weights=(0.5, 1.0))
        @test_throws ArgumentError MultiplicativeUtility(linear; weights=(0.5, 0.0))
        @test_throws ArgumentError MultiplicativeUtility((LinearUtility(:a), linear[2]);
                                                         weights=(0.5, 0.5))
        @test_throws ArgumentError certainty_equivalent(u, [1, 2])
        @test_throws ArgumentError inverse_utility(u, 0.5)
    end
end
