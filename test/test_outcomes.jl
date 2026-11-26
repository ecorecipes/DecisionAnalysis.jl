# Outcome tables, the driver with common random numbers, utilities and payoff tables.

# A toy simulator, defined here: a pond of frogs with a parameter-dependent survival
# and an action that adds protection. It records the first random number it draws, so
# the test can see the common random numbers.
struct ToyPond
    frogs::Int
    survival::Float64
    protection::Float64
    fail::Bool
end
function DecisionAnalysis.with_parameters(p::ToyPond, θ)
    return ToyPond(θ.frogs, θ.survival,
                   p.protection, p.fail)
end
function DecisionAnalysis.apply_action(p::ToyPond, a::Symbol)
    a === :none && return p
    a === :fence && return ToyPond(p.frogs, p.survival, 0.2, p.fail)
    a === :broken && return ToyPond(p.frogs, p.survival, 0.0, true)
    return throw(ArgumentError("unknown action $a"))
end
function DecisionAnalysis.simulate(p::ToyPond; rng)
    p.fail && throw(DomainError(p.frogs, "the toy model failed"))
    u = rand(rng)
    alive = count(_ -> rand(rng) < min(p.survival + p.protection, 1.0), 1:(p.frogs))
    infected = count(_ -> rand(rng) < 0.3, 1:alive)
    return (; u, alive, infected)
end
function DecisionAnalysis.outcome(sol::NamedTuple)
    return (; uninfected=sol.alive - sol.infected, infected=sol.infected, first_draw=sol.u)
end

# A second toy, deterministic, whose solution is a plain vector (a type it does not own)
# and whose outcomes depend on the problem (its horizon): it gives `outcome` a method of
# the two-argument form, on the problem, and none for the vector.
struct ToyGrowth
    rate::Float64
    harvest::Float64
    horizon::Int
end
DecisionAnalysis.with_parameters(p::ToyGrowth, θ) = ToyGrowth(θ.rate, p.harvest, p.horizon)
DecisionAnalysis.apply_action(p::ToyGrowth, a::Real) = ToyGrowth(p.rate, a, p.horizon)
function DecisionAnalysis.simulate(p::ToyGrowth; rng)
    return accumulate((n, _) -> n * p.rate * (1 - p.harvest), 1:10; init=10.0)
end
function DecisionAnalysis.outcome(p::ToyGrowth, path::Vector{Float64})
    return (; final=path[p.horizon], total=sum(path[1:(p.horizon)]))
end

@testset "outcome tables" begin
    @testset "from a table of runs" begin
        runs = NamedTuple[]
        for d in ("p3", "p1", "p2"), a in ("none", "treat"), r in 1:4
            status = (d == "p1" && a == "treat" && r == 2) ? "error:Boom" : "ok"
            push!(runs,
                  (; point=d, arm=a, rep=r, status,
                   uninfected=status == "ok" ? 10 * r + (a == "treat") * 5 : -1,
                   infected=status == "ok" ? r : -1, group=d == "p3" ? "B" : "A"))
        end
        parameters = [(; point="p1", growth=0.1, weight=2.0),
                      (; point="p2", growth=0.2, weight=1.0),
                      (; point="p3", growth=0.3, weight=1.0)]
        t = OutcomeTable(runs; draw=:point, action=:arm, replicate=:rep,
                         outcomes=[:uninfected, :infected], status=:status, parameters,
                         weights=:weight)
        # The draws take the order of the parameter table, not of the runs.
        @test state_labels(t) == [:p1, :p2, :p3]
        @test action_labels(t) == [:none, :treat]
        @test outcome_names(t) == [:uninfected, :infected]
        @test draw_weights(t) == [0.5, 0.25, 0.25]
        @test parameter_draws(t) == (; growth=[0.1, 0.2, 0.3])
        # The failed run is kept and counted, not dropped.
        @test run_counts(t) == (; runs=24, ok=23, failed=1, absent=0)
        @test failed_runs(t) ==
              [(; action=:treat, draw=:p1, replicate=Symbol("2"), status="error:Boom")]
        @test replicate_counts(t) == [4 4 4; 3 4 4]
        @test occursin("1 failed", sprint(show, t))
        b = SampleBelief(t)
        @test state_labels(b) == state_labels(t) && probabilities(b) == draw_weights(t)

        # Slow oracle for the payoff table: explicit loops over the rows.
        u = InfectionUtility(; weight=2)
        payoff = payoff_table(t, u)
        for (i, a) in enumerate(("none", "treat")), (j, d) in enumerate(("p1", "p2", "p3"))
            us = [row.uninfected - 2 * row.infected
                  for row in runs if row.point == d && row.arm == a && row.status == "ok"]
            m, se = oracle_cell(us)
            @test payoff.Q[i, j] == sum(us) // length(us)
            @test payoff.standard_errors[i, j] ≈ se
            @test payoff.replicates[i, j] == length(us)
        end
        @test payoff.Q isa Matrix{Rational{Int}}
        @test !single_replicate(payoff)
        # Standard errors are a separate field: the payoffs are plain means.
        @test payoff.Q[1, 1] == (10 + 20 + 30 + 40 - 2 * (1 + 2 + 3 + 4)) // 4

        # Gains over the baseline pair the replicates: the failed replicate is left out
        # of the pair, and the paired standard error is that of the differences.
        gains = payoff_table(t, u; baseline=:none)
        @test all(iszero, gains.Q[1, :]) && all(iszero, gains.standard_errors[1, :])
        @test gains.Q[2, :] == [5, 5, 5]
        @test all(iszero, gains.standard_errors[2, :])   # the difference is constant
        @test gains.replicates == [4 4 4; 3 4 4]
        @test_throws InvalidBaselineError payoff_table(t, u; baseline=:nope)

        # Parameter columns carried in the runs.
        t2 = OutcomeTable(runs; draw=:point, action=:arm, replicate=:rep,
                          outcomes=[:uninfected], status=:status, parameters=[:group])
        @test state_labels(t2) == [:p3, :p1, :p2]
        @test parameter_draws(t2) == (; group=["B", "A", "A"])
        @test draw_weights(t2) == fill(1 // 3, 3)
        t3 = OutcomeTable(runs; draw=:point, action=:arm, replicate=:rep,
                          outcomes=[:uninfected], status=:status, actions=[:treat, :none])
        @test action_labels(t3) == [:treat, :none]
        @test payoff_table(t3, IndividualsUtility(:uninfected)).Q[2, :] ==
              payoff_table(t2, IndividualsUtility(:uninfected)).Q[1, :]

        # The same utility through the three doors.
        @test individuals_payoff(t, :uninfected) ==
              payoff_table(t, IndividualsUtility(:uninfected))
        @test infection_payoff(t; weight=2) == payoff
        un = individuals_payoff(t, :uninfected)
        inf = individuals_payoff(t, :infected)
        @test infection_payoff(; uninfected=un, infected=inf, weight=2).Q == payoff.Q
        @test payoff_table(t, o -> o.uninfected - 2 * o.infected).Q == payoff.Q
        @test payoff_table(t,
                           FunctionUtility(o -> o.uninfected / (o.uninfected + o.infected))).Q isa
              Matrix{Float64}
        @test u((; uninfected=7, infected=2)) == 3
        @test IndividualsUtility(:n)((; n=4)) == 4
        @test FunctionUtility(o -> 2 * o.n)((; n=4)) == 8

        problem = DecisionProblem(t, u; baseline=:none)
        @test problem.belief == b && problem.payoff == payoff
        @test first_choice(decide(problem, MaxExpectedUtility())) == :treat
    end

    @testset "errors and flags" begin
        row(d, a, r; n=1, status="ok") = (; draw=d, action=a, replicate=r, n, status)
        @test_throws InvalidOutcomeError OutcomeTable([row(1, :a, 1), row(1, :a, 1)];
                                                      outcomes=[:n])
        @test_throws InvalidOutcomeError OutcomeTable([row(1, :a, 1; n=NaN)];
                                                      outcomes=[:n])
        @test_throws InvalidOutcomeError OutcomeTable([row(1, :a, 1; n=missing)];
                                                      outcomes=[:n])
        @test_throws InvalidOutcomeError OutcomeTable([row(1, :a, 1; status="bad")];
                                                      outcomes=[:n], status=:status)
        @test_throws ArgumentError OutcomeTable([row(1, :a, 1)]; outcomes=[:m])
        @test_throws ArgumentError OutcomeTable([row(1, :a, 1)]; outcomes=Symbol[])
        @test_throws ArgumentError OutcomeTable([row(1, :a, 1)]; outcomes=[:n],
                                                actions=[:b])
        @test_throws ArgumentError OutcomeTable([row(1, :a, 1)]; outcomes=[:n],
                                                weights=:w)
        @test_throws ArgumentError OutcomeTable(3; outcomes=[:n])
        @test_throws InvalidOutcomeError OutcomeTable([row(1, :a, 1), row(2, :a, 1)];
                                                      outcomes=[:n],
                                                      parameters=[(; draw=1, x=1.0)])
        @test_throws InvalidOutcomeError OutcomeTable([(; draw=1, action=:a, replicate=r,
                                                        n=1, g=r) for r in 1:2];
                                                      outcomes=[:n], parameters=[:g])
        # A NaN in a failed run is not an error: the run is not used.
        t = OutcomeTable([row(1, :a, 1), row(1, :a, 2; n=NaN, status="crashed"),
                          row(1, :b, 1; n=NaN, status="crashed"), row(1, :b, 2; n=3)];
                         outcomes=[:n], status=:status)
        @test run_counts(t) == (; runs=4, ok=2, failed=2, absent=0)
        p = payoff_table(t, IndividualsUtility(:n))
        @test p.Q == [1; 3;;] && p.replicates == [1; 1;;]
        # One replicate per cell: flagged, and no standard error is invented.
        @test single_replicate(p)
        @test p.standard_errors === nothing
        @test occursin("single simulator run", sprint(show, MIME"text/plain"(), p))
        @test !single_replicate(PayoffTable([1 2; 3 4]))
        # No usable run in a cell: refused when a payoff is asked.
        t0 = OutcomeTable([row(1, :a, 1), row(1, :b, 1; status="crashed")];
                          outcomes=[:n], status=:status)
        @test_throws InvalidOutcomeError payoff_table(t0, IndividualsUtility(:n))
        @test_throws InvalidOutcomeError payoff_table(t, IndividualsUtility(:nope))
        @test_throws InvalidOutcomeError payoff_table(OutcomeTable([row(1, :a, 1; n=-2)];
                                                                   outcomes=[:n]),
                                                      IndividualsUtility(:n))
        @test_throws InvalidOutcomeError payoff_table(t, o -> NaN)
        @test_throws ArgumentError InfectionUtility(; weight=-1)
        @test_throws InvalidPayoffError PayoffTable([1 2]; replicates=[0 1])
        @test_throws InvalidPayoffError PayoffTable([1 2]; replicates=[1 1 1])
        # Absent cells are counted.
        ta = OutcomeTable([row(1, :a, 1), row(1, :a, 2), row(1, :b, 1)]; outcomes=[:n])
        @test run_counts(ta) == (; runs=3, ok=3, failed=0, absent=1)
        # Replicate counts travel with a table and drop out of a sum.
        @test (2 * p).replicates == p.replicates
        @test (p + p).replicates === nothing
    end

    @testset "the driver, with common random numbers" begin
        draws = (; frogs=[20, 40, 60], survival=[0.3, 0.5, 0.7])
        actions = [:none, :fence]
        t = outcomes(ToyPond(0, 0.0, 0.0, false), actions, draws; replicates=5, seed=7)
        @test state_labels(t) == [:s1, :s2, :s3]
        @test action_labels(t) == [:none, :fence]
        @test outcome_names(t) == [:uninfected, :infected, :first_draw]
        @test run_counts(t) == (; runs=30, ok=30, failed=0, absent=0)
        @test parameter_draws(t) == draws
        # Common random numbers: the first draw of the generator is the same for both
        # actions in every (draw, replicate), and differs between replicates and draws.
        first_draw = t.values[:, :, :, 3]
        @test first_draw[1, :, :] == first_draw[2, :, :]
        @test allunique(first_draw[1, :, :])
        # Reproducible, and the seed matters.
        again = outcomes(ToyPond(0, 0.0, 0.0, false), actions, draws; replicates=5, seed=7)
        @test again.values == t.values
        other = outcomes(ToyPond(0, 0.0, 0.0, false), actions, draws; replicates=5, seed=8)
        @test other.values != t.values
        # The fence helps in every cell, as the model says it must on average.
        p = DecisionProblem(t, o -> o.uninfected + o.infected; baseline=:none)
        @test all(>(0), expected_gain_over_baseline(p)[2:2])

        # A weighted sample goes through with its weights and labels.
        b = SampleBelief((; id=["x", "y"], frogs=[10, 30], survival=[0.4, 0.6], w=[1, 3]);
                         weights=:w, labels=:id)
        tb = outcomes(ToyPond(0, 0.0, 0.0, false), (; none=:none, fence=:fence), b;
                      replicates=2, seed=1)
        @test state_labels(tb) == [:x, :y] && draw_weights(tb) == [1 // 4, 3 // 4]

        # A run that throws is a failed run with its status, not a missing row.
        tf = outcomes(ToyPond(0, 0.0, 0.0, false), [:none, :broken], draws; replicates=2,
                      seed=7)
        @test run_counts(tf) == (; runs=12, ok=6, failed=6, absent=0)
        @test all(f -> f.action == :broken && f.status == "error:DomainError",
                  failed_runs(tf))
        @test_throws InvalidOutcomeError payoff_table(tf, IndividualsUtility(:uninfected))
        @test_throws InvalidOutcomeError outcomes(ToyPond(0, 0.0, 0.0, false), [:broken],
                                                  draws; seed=7)
        @test_throws ArgumentError outcomes(ToyPond(0, 0.0, 0.0, false), actions, draws)
        @test_throws ArgumentError outcomes(ToyPond(0, 0.0, 0.0, false), actions, draws;
                                            seed=1, replicates=0)
        @test_throws ArgumentError outcomes(ToyPond(0, 0.0, 0.0, false), Symbol[], draws;
                                            seed=1)
        # The adapter verbs have no methods for foreign types.
        @test_throws MethodError with_parameters(1.0, (;))
        @test_throws MethodError apply_action(1.0, :a)
        @test_throws MethodError simulate(1.0; rng=Xoshiro(1))
        @test_throws MethodError outcome(1.0)
        @test_throws MethodError parameter_draws(1.0)
        @test_throws MethodError draw_weights(1.0)

        # `on_error=:throw` rethrows the first exception of a run instead of recording it.
        @test_throws DomainError outcomes(ToyPond(0, 0.0, 0.0, false), [:none, :broken],
                                          draws; seed=7, on_error=:throw)
        @test outcomes(ToyPond(0, 0.0, 0.0, false), actions, draws; replicates=5, seed=7,
                       on_error=:throw).values == t.values
        @test_throws ArgumentError outcomes(ToyPond(0, 0.0, 0.0, false), actions, draws;
                                            seed=7, on_error=:ignore)
    end

    @testset "the model's own seed convention" begin
        pond = ToyPond(0, 0.0, 0.0, false)
        draws = (; id=[11, 12, 13], frogs=[20, 40, 60], survival=[0.3, 0.5, 0.7])
        convention(θ, r) = UInt64(1000 * θ.id + r)
        t = outcomes(pond, [:none, :fence], draws; replicates=4, run_seed=convention)
        # Every run used the generator the convention names, the same for both actions.
        for d in 1:3, r in 1:4
            θ = (; id=draws.id[d], frogs=draws.frogs[d], survival=draws.survival[d])
            u = rand(Xoshiro(convention(θ, r)))
            @test t.values[1, d, r, 3] == u && t.values[2, d, r, 3] == u
        end
        # So runs made elsewhere by the same convention are reproduced exactly, and a
        # new action shares random numbers with them: the two go in one table.
        stored = [(; draw=draws.id[d], action=:none, replicate=r,
                   DecisionAnalysis.outcome(DecisionAnalysis.simulate(DecisionAnalysis.with_parameters(pond,
                                                                                                       (;
                                                                                                        frogs=draws.frogs[d],
                                                                                                        survival=draws.survival[d]));
                                                                      rng=Xoshiro(convention((;
                                                                                              id=draws.id[d]),
                                                                                             r))))...)
                  for d in 1:3 for r in 1:4]
        old = OutcomeTable(stored; outcomes=[:uninfected, :infected, :first_draw])
        new = outcomes(pond, [:fence], draws; replicates=4, run_seed=convention)
        @test old.values[1, :, :, :] == t.values[1, :, :, :]
        @test new.values[1, :, :, :] == t.values[2, :, :, :]
        @test new.values[1, :, :, 3] == old.values[1, :, :, 3]
        # The stored runs relabelled as the driver labels them, joined with the new
        # action: the table of one call with both actions.
        relabelled = OutcomeTable([(; row..., replicate="r$(row.replicate)")
                                   for row in stored];
                                  outcomes=[:uninfected, :infected, :first_draw],
                                  parameters=draws, key=:id)
        fresh = outcomes(pond, [:fence],
                         SampleBelief(draws; labels=draws.id); replicates=4,
                         run_seed=convention)
        joined = combine_actions(relabelled, fresh)
        @test action_labels(joined) == [:none, :fence]
        @test joined.values == t.values && all(==("ok"), joined.status)
        @test state_labels(joined) == [Symbol(11), Symbol(12), Symbol(13)]
        @test crn_correlation(joined, IndividualsUtility(:first_draw)) ≈ 1
        # One of `seed` and `run_seed`; integer seeds; no two runs alike.
        @test_throws ArgumentError outcomes(pond, [:none], draws; seed=1,
                                            run_seed=convention)
        @test_throws ArgumentError outcomes(pond, [:none], draws; run_seed=(θ, r) -> 0.5)
        @test_throws ArgumentError outcomes(pond, [:none], draws; replicates=2,
                                            run_seed=(θ, r) -> θ.id)
    end

    @testset "combining the actions of several tables" begin
        row(d, a, r; n=10d + r, m=d, status="ok") = (; draw=d, action=a, replicate=r, n,
                                                     m, status)
        pars = [(; draw=d, x=d / 2, w=d) for d in 1:3]
        build(rows; kw...) = OutcomeTable(rows; outcomes=[:n, :m], status=:status,
                                          parameters=pars, weights=:w, kw...)
        a = build([row(d, :a, r) for d in 1:3 for r in 1:2])
        # The second table lists its draws and replicates in another order, records
        # its outcomes in another order, and has a failed run.
        brows = [row(d, act, r; n=100d + r, status=(d == 2 && r == 1) ? "crashed" : "ok")
                 for r in (2, 1) for d in (3, 1, 2) for act in (:b, :c)]
        b = OutcomeTable(brows; outcomes=[:m, :n], status=:status,
                         parameters=[(; draw=d, x=d / 2, w=d) for d in (3, 1, 2)],
                         weights=:w)
        t = combine_actions(a, b)
        @test action_labels(t) == [:a, :b, :c] && outcome_names(t) == [:n, :m]
        @test state_labels(t) == state_labels(a) && draw_weights(t) == draw_weights(a)
        @test parameter_draws(t) == parameter_draws(a)
        @test run_counts(t) == (; runs=18, ok=16, failed=2, absent=0)
        # Against the table built from all the rows at once.
        whole = build(vcat([row(d, :a, r) for d in 1:3 for r in 1:2],
                           sort(brows; by=x -> (x.draw, x.replicate)));
                      actions=[:a, :b, :c])
        @test t.status == whole.status
        ok = t.status .== "ok"
        @test all(t.values[:, :, :, k][ok] == whole.values[:, :, :, k][ok] for k in 1:2)
        @test payoff_table(t, IndividualsUtility(:n)).Q ==
              payoff_table(whole, IndividualsUtility(:n)).Q
        @test combine_actions(a).values == a.values
        # Only the outcomes every table records are kept.
        c = OutcomeTable([row(d, :c, r) for d in 1:3 for r in 1:2]; outcomes=[:m],
                         parameters=pars, weights=:w)
        @test outcome_names(combine_actions(a, c)) == [:m]
        # What must agree.
        @test_throws ArgumentError combine_actions(a, a)
        other(rows; kw...) = OutcomeTable(rows; outcomes=[:n, :m], kw...)
        @test_throws ArgumentError combine_actions(a,
                                                   other([row(d, :b, r) for d in 1:2
                                                          for r in 1:2];
                                                         parameters=pars[1:2]))
        @test_throws ArgumentError combine_actions(a,
                                                   build([row(d, :b, r) for d in 1:3
                                                          for r in 1:3]))
        @test_throws ArgumentError combine_actions(a,
                                                   other([row(d, :b, r) for d in 1:3
                                                          for r in 1:2];
                                                         parameters=pars))
        @test_throws ArgumentError combine_actions(a,
                                                   other([row(d, :b, r) for d in 1:3
                                                          for r in 1:2];
                                                         parameters=[(; draw=d, x=1.0 * d,
                                                                      w=d) for d in 1:3],
                                                         weights=:w))
        @test_throws ArgumentError combine_actions(a,
                                                   OutcomeTable([(; draw=d, action=:b,
                                                                  replicate=r, z=1)
                                                                 for d in 1:3 for r in 1:2];
                                                                outcomes=[:z],
                                                                parameters=pars,
                                                                weights=:w))
    end

    @testset "single_replicate of an outcome table" begin
        pond = ToyPond(0, 0.0, 0.0, false)
        draws = (; frogs=[20, 40], survival=[0.3, 0.5])
        one = outcomes(pond, [:none, :fence], draws; seed=3)
        many = outcomes(pond, [:none, :fence], draws; replicates=3, seed=3)
        @test single_replicate(one) && !single_replicate(many)
        for t in (one, many)
            @test single_replicate(t) ==
                  single_replicate(payoff_table(t, IndividualsUtility(:uninfected)))
        end
        # A cell left with one usable run by a failure is flagged too.
        runs = [(; draw=1, action=a, replicate=r, n=r,
                 status=(a == :b && r == 2) ? "x" : "ok")
                for a in (:a, :b) for r in 1:2]
        @test single_replicate(OutcomeTable(runs; outcomes=[:n], status=:status))
    end

    @testset "outcomes read with the problem; a deterministic model" begin
        # `outcome(prob, solution)` is called by the driver; its fallback is
        # `outcome(solution)`, which the first toy relies on.
        @test outcome(ToyPond(0, 0.0, 0.0, false), (; u=0.5, alive=7, infected=2)) ==
              (; uninfected=5, infected=2, first_draw=0.5)
        @test_throws MethodError outcome(ToyGrowth(1.0, 0.0, 3), 1.0)
        @test_throws MethodError outcome([1.0, 2.0])       # no method for the foreign type
        rates = (; rate=[1.1, 0.9, 1.3])
        harvests = [0.0, 0.1, 0.3]
        t = outcomes(ToyGrowth(1.0, 0.0, 4), harvests, rates; replicates=3, seed=1)
        @test outcome_names(t) == [:final, :total]
        for (a, h) in enumerate(harvests), (d, r) in enumerate(rates.rate)
            path = [10.0 * (r * (1 - h))^k for k in 1:4]      # the model, by hand
            @test all(isapprox.(t.values[a, d, :, 1], path[4]; rtol=1e-12))
            @test all(isapprox.(t.values[a, d, :, 2], sum(path); rtol=1e-12))
        end
        # The horizon is the problem's: another problem reads other outcomes of the same runs.
        t6 = outcomes(ToyGrowth(1.0, 0.0, 6), harvests, rates; seed=1)
        @test t6.values[1, 1, 1, 1] ≈ 10.0 * 1.1^6
        # Equal replicates have a standard error of exactly zero, also for paired gains.
        for baseline in (nothing, Symbol("0.0"))
            payoff = payoff_table(t, o -> o.total; baseline)
            @test payoff.standard_errors !== nothing
            @test all(==(0.0), payoff.standard_errors)
        end
    end
end
