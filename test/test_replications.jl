# Replications of published value-of-information analyses. Each fixture under
# test/fixtures/replications/ holds the small input table of one paper, the published
# numbers and their provenance; the tests assert our result against the published one
# to the precision printed, and pin every place where the two differ together with the
# reason found (the fixture's "resolution" fields). Nothing here is fitted: a failure
# means that the package, or the transcription, has changed.

using LinearAlgebra: eigvals
using Statistics: quantile

const REPLICATIONS = joinpath(@__DIR__, "fixtures", "replications")
replication(name) = JSON.parse(read(joinpath(REPLICATIONS, name * ".json"), String))

# "84.5" -> 169//2 and "1/3" -> 1//3, exactly.
function exact_number(s::AbstractString)
    occursin('/', s) && return parse_rational(s)
    parts = split(s, '.')
    length(parts) == 1 && return parse(BigInt, s) // 1
    sign = startswith(s, "-") ? -1 : 1
    whole = abs(parse(BigInt, parts[1]))
    return sign * (whole + parse(BigInt, parts[2]) // big(10)^length(parts[2]))
end
exact_number(x::Integer) = BigInt(x) // 1
exact_vector(v) = BigRat[exact_number(x) for x in v]
target(fix, id) = only(t for t in fix["targets"] if t["id"] == id)
function agrees(ours, fix, id)
    t = target(fix, id)
    return abs(Float64(ours) - t["value"]) <= t["tolerance"] + 1e-9
end

@testset "replications of published analyses" begin
    @testset "every fixture states its provenance" begin
        files = filter(endswith(".json"), readdir(REPLICATIONS))
        @test length(files) == 5
        for f in files
            fix = JSON.parse(read(joinpath(REPLICATIONS, f), String))
            for key in ("citation", "doi", "licence", "retrieved", "decision", "unit",
                        "baseline")
                @test haskey(fix, key) && !isempty(fix[key])
            end
            @test fix["name"] * ".json" == f
        end
    end

    @testset "Shea et al. 2014, foot-and-mouth disease" begin
        fix = replication("shea2014_fmd")
        actions, states = symbols(fix["actions"]), symbols(fix["states"])
        printed = reduce(hcat,
                         [exact_vector(fix["table2"]["cost_as_printed"][s])
                          for s in fix["states"]])
        cost = copy(printed)
        fixup = fix["table2"]["correction"]
        i = findfirst(==(Symbol(fixup["action"])), actions)
        j = findfirst(==(Symbol(fixup["state"])), states)
        @test printed[i, j] == exact_number(fixup["printed"])
        cost[i, j] = exact_number(fixup["corrected"])
        for case in ("case1", "case2")
            w = rational_vector(fix["weights"][case])
            problem = DecisionProblem(PayoffTable(actions, states, -cost),
                                      FiniteBelief(states, w); baseline=:IP)
            # The printed weighted averages, with the corrected entry ...
            averages = Float64.(cost * w)
            shown = Float64.(exact_vector(fix["table2"]["printed_weighted_average"][case]))
            @test maximum(abs, averages .- shown) <= 0.1
            @test first_choice(decide(problem, MaxExpectedUtility())) == :CP
            @test agrees(-act_now_value(problem), fix, "best_fixed_" * case)
            @test agrees(-act_now_value(problem) - evpi(problem), fix, "perfect_" * case)
            @test agrees(evpi(problem), fix, "evpi_" * case)
            @test agrees(100 * evpi(problem) / -act_now_value(problem), fix,
                         "evpi_percent_" * case)
            @test evpi(problem) ==
                  oracle_evpi(-cost, w) ==
                  expected_regret(problem)[findfirst(==(:CP), actions)]
            # ... and not with the entry as printed: the misprint is in Table 2.
            asprinted = DecisionProblem(PayoffTable(actions, states, -printed),
                                        FiniteBelief(states, w))
            @test abs(Float64((printed * w)[i]) - shown[i]) > 39
            @test Float64(evpi(asprinted)) < 9
        end

        # The two-stage adaptive policy, from the sixteen strategies of Table S1.
        stages = symbols(fix["tableS1"]["stages"])
        courses = [Symbol(a, "_then_", b) for a in stages for b in stages]
        first_stage = [a for a in stages for b in stages]
        fixed = [Symbol(a, "_then_", a) for a in stages]
        S1 = reduce(hcat, [BigRat.(fix["tableS1"]["cost"][s]) for s in fix["states"]])
        S2 = reduce(hcat,
                    [exact_vector(fix["tableS2"]["first_stage_cost"][s])
                     for s in fix["states"]])
        # Table S2 is the row minimum of Table S1 within each first stage.
        for (g, stage) in enumerate(stages), k in eachindex(states)
            best = minimum(S1[r, k] for r in eachindex(courses) if first_stage[r] == stage)
            @test abs(best - S2[g, k]) <= 1 // 2
        end
        # The fixed strategies of Table S1 are Table 2 rounded (K3 under CP is 378).
        rows = [findfirst(==(c), courses) for c in fixed]
        @test maximum(abs, S1[rows, :] .- cost) <= 1 // 2
        for (case, stage) in (("case1", :CP), ("case2", :DC))
            w = rational_vector(fix["weights"][case])
            belief = FiniteBelief(states, w)
            problem = DecisionProblem(PayoffTable(courses, states, -S1), belief;
                                      feasible=a -> true)
            r = two_stage_value(problem, first_stage)
            @test r.chosen == [stage]
            @test agrees(-r.value, fix, "adaptive_cost_" * case)
            shown = Float64.(exact_vector(fix["tableS2"]["printed_weighted_average"][case]))
            @test maximum(abs, Float64.(-r.values) .- shown) <= 0.6
            # The same from the one-decimal Table S2, and against the best fixed
            # strategy of Table 2.
            exactS2 = minimum(S2 * w)
            @test agrees(exactS2, fix, "adaptive_cost_" * case)
            best_fixed = minimum(cost * w)
            evpi_fixed = best_fixed - sum(w .* [minimum(cost[:, k]) for k in 1:3])
            gain = case == "case1" ? best_fixed - (-r.value) : best_fixed - exactS2
            @test agrees(gain, fix, "adaptive_gain_" * case)
            case == "case2" &&
                @test agrees(100 * gain / evpi_fixed, fix, "adaptive_share_case2_percent")
            # No switch of policy is ever worse than the best fixed strategy here, and
            # learning after a month cannot beat learning before the outbreak.
            @test r.act_now <= r.value <= r.act_now + evpi(problem)
            # Slow oracle for the two-stage value.
            slow = maximum(stages) do g
                return sum(eachindex(states)) do k
                    return w[k] * maximum(-S1[r, k] for r in eachindex(courses)
                                          if first_stage[r] == g)
                end
            end
            @test slow == r.value
        end
        # Figure 3D: the share of EVPI recovered against the weight on the UK kernel,
        # the rest split evenly.
        share = function (u)
            w = [(1 - u) / 2, u, (1 - u) / 2]
            problem = DecisionProblem(PayoffTable(courses, states, -S1),
                                      FiniteBelief(states, w))
            r = two_stage_value(problem, first_stage)
            return r.gain / evpi(problem)
        end
        lo = target(fix, "share_above_80_percent_when_UK_weight_below")
        hi = target(fix, "share_below_2_percent_when_UK_weight_above")
        @test (lo["value"], hi["value"]) == (0.13, 0.41)
        @test share(12 // 100) > 4 // 5 > share(14 // 100)
        # From the whole-number Table S1 the 2% point is at 0.43, not 0.41.
        @test share(40 // 100) > 1 // 50 > share(44 // 100)
        @test share(1 // 2) < 1 // 50 < share(3 // 10)
    end

    @testset "Runge et al. 2011, whooping cranes" begin
        fix = replication("runge2011_cranes")
        actions, states = symbols(fix["actions"]), symbols(fix["states"])
        raw = exact_vector(fix["weights_percent"])
        @test sum(raw) == 1001 // 10
        w = raw ./ sum(raw)
        score = permutedims(reduce(hcat, [exact_vector(r) for r in fix["table5"]["score"]]))
        @test size(score) == (8, 7)
        problem = DecisionProblem(PayoffTable(actions, states, permutedims(score)),
                                  FiniteBelief(states, w); baseline=:wait)
        shown = Float64.(exact_vector(fix["table5"]["printed_expected_value"]))
        @test maximum(abs, Float64.(aggregate(Expectation(), problem)) .- shown) <= 0.001
        @test first_choice(decide(problem, MaxExpectedUtility())) == :restore_meadows
        @test agrees(act_now_value(problem), fix, "act_now")
        @test agrees(act_now_value(problem) + evpi(problem), fix, "perfect")
        @test agrees(evpi(problem), fix, "evpi")
        @test agrees(100 * evpi(problem) / act_now_value(problem), fix, "evpi_percent")
        @test evpi(problem) == oracle_evpi(permutedims(score), w)

        # "EVPXI": the value of confirming or refuting one hypothesis.
        partial = Dict(h => evppi(problem, hypothesis_partition(problem, h))
                       for h in states)
        t = target(fix, "evpxi_share_percent")
        for (h, percent) in t["value"]
            @test abs(100 * Float64(partial[Symbol(h)] / evpi(problem)) - percent) <=
                  t["tolerance"]
        end
        for h in setdiff(states, Symbol.(keys(t["value"])))
            @test partial[h] < evpi(problem) / 100
        end
        # They add up to EVPI here, exactly, which is not a general property.
        @test sum(values(partial)) == evpi(problem)
        # Slow oracle for one of them: weights renormalised when the hypothesis is refuted.
        k = findfirst(==(:black_flies), states)
        rest = setdiff(eachindex(states), k)
        refuted = w[rest] ./ sum(w[rest])
        informed = w[k] * maximum(score[k, :]) +
                   (1 - w[k]) * maximum(permutedims(score[rest, :]) * refuted)
        @test informed - act_now_value(problem) == partial[:black_flies]

        # Fledging rate alone (Table 2): the best action under uncertainty is best under
        # no single hypothesis.
        fledge = permutedims(reduce(hcat,
                                    [exact_vector(r)
                                     for r in fix["table2"]["fledging_rate"]]))
        fproblem = DecisionProblem(PayoffTable(actions, states, permutedims(fledge)),
                                   FiniteBelief(states, w); baseline=:wait)
        @test first_choice(decide(fproblem, MaxExpectedUtility())) == :restore_meadows
        @test all(argmax(fledge[h, :]) != 4 for h in 1:8)
        @test agrees(act_now_value(fproblem), fix, "fledging_act_now")
        @test agrees(act_now_value(fproblem) + evpi(fproblem), fix, "fledging_perfect")
        @test agrees(100 * evpi(fproblem) / act_now_value(fproblem), fix,
                     "fledging_percent")

        # Learning by doing: observe the weighted score of one action with normal error.
        Q = Float64.(permutedims(score))
        fp = DecisionProblem(PayoffTable(actions, states, Q), FiniteBelief(states, Float64.(w)))
        total = evpi(fp)
        fraction(a, sd; bins=1200) = evsi(fp,
                                          gaussian_experiment(Q[findfirst(==(a), actions), :],
                                                              sd; states, bins)) / total
        for r in fix["evsi"]["figure_readings"]
            @test abs(fraction(Symbol(r["action"]), r["sd"]) - r["fraction"]) <=
                  r["tolerance"]
        end
        best(sd) = actions[argmax([fraction(a, sd) for a in actions])]
        @test best(0.02) == :kill_flies
        @test fraction(:kill_flies, 0.02) > 0.6
        @test best(0.04) == :no_salvage == best(0.1)
        @test sort(actions; by=a -> -fraction(a, 0.06))[2] == :swap_eggs
        @test fraction(:restore_meadows, 0.03) < 0.3
        # Where our numbers leave the paper's sentence (see the fixture's resolution).
        @test 0.49 < fraction(:kill_flies, 0.03) < 0.52
        @test best(0.03) == :no_salvage && best(0.028) == :kill_flies
        @test fraction(:kill_flies, 0.023) > 0.6 > fraction(:kill_flies, 0.024)
        # The discretised experiment against a Monte Carlo oracle that shares no code
        # with it (posterior form, 200,000 draws), and convergence in the bins.
        rng = Random.Xoshiro(20261123)
        p, mu, sd = Float64.(w), Q[2, :], 0.03
        acc = 0.0
        n = 200_000
        for _ in 1:n
            s = findfirst(>=(rand(rng)), cumsum(p))
            s === nothing && (s = 8)
            x = mu[s] + sd * randn(rng)
            post = p .* exp.(-abs2.((x .- mu) ./ sd) ./ 2)
            post ./= sum(post)
            acc += maximum(Q * post)
        end
        @test abs((acc / n - maximum(Q * p)) / total - fraction(:kill_flies, 0.03)) < 0.01
        @test abs(fraction(:kill_flies, 0.03; bins=2400) - fraction(:kill_flies, 0.03)) <
              1e-4
        # A measurement can never be worth more than the state, and a sharper one is
        # worth at least as much.
        @test 0 <= fraction(:wait, 0.2) <= fraction(:wait, 0.05) <=
              fraction(:wait, 0.005) <= 1
    end

    @testset "Eze et al. 2023, ICU beds" begin
        fix = replication("eze2023_icu")
        actions = symbols(fix["actions"])
        beds = fix["beds_added"]
        X = BigRat.(fix["table4"]["shortfall"])
        levels = fix["levels"]
        triples = [(r, c, i) for r in 1:3 for c in 1:3 for i in 1:3]
        states = [Symbol(levels[r], "_", levels[c], "_", levels[i]) for (r, c, i) in triples]
        @test length(X) == 27 == length(unique(states))
        joint(m) = [m[r] * m[c] * m[i] for (r, c, i) in triples]
        loss(x, b, gamma) = x >= b ? x - b : gamma * (b - x)
        table(x, gamma) = [-loss(xs, b, gamma) for b in beds, xs in x]
        gammas = rational_vector(fix["gammas"])
        nb = fix["notebook"]
        tol = nb["tolerance"]
        problem(belief, gamma) = DecisionProblem(PayoffTable(actions, states,
                                                             table(X, gamma)),
                                                 FiniteBelief(states,
                                                              joint(rational_vector(fix["beliefs"][belief])));
                                                 baseline=:add_0)
        # EVPI against the weight on excess beds, three beliefs (authors' notebook).
        for (belief, curve) in nb["evpi"]
            ours = [Float64(evpi(problem(belief, g))) for g in gammas]
            @test maximum(abs, ours .- curve) <= tol
        end
        @test all(iszero(evpi(problem(b, 0 // 1))) for b in ("none", "low", "medium", "severe"))
        # The medium curve was printed from a prior that sums to 0.99.
        m = joint(rational_vector(fix["beliefs"]["medium"]))
        slip = nb["medium_misprint"]
        @test m[slip["scenario"]] == parse_rational(slip["intended"])
        bad = copy(m)
        bad[slip["scenario"]] = parse_rational(slip["used"])
        @test sum(bad) == 99 // 100
        asprinted = [Float64(oracle_evpi(table(X, g), bad)) for g in gammas]
        @test maximum(abs, asprinted .- nb["evpi_medium_as_printed"]) <= tol
        proper = [Float64(evpi(problem("medium", g))) for g in gammas]
        @test 0.2 < maximum(abs, proper .- nb["evpi_medium_as_printed"]) < 0.4
        # The article's text.
        text = fix["article_text"]
        @test abs(Float64(evpi(problem("none", 4 // 5))) -
                  text["evpi_none_gamma08"]["value"]) <= text["evpi_none_gamma08"]["tolerance"]
        @test maximum(Float64(evpi(problem("low", g))) for g in gammas) < 45.3    # not 33
        @test abs(Float64(evpi(problem("severe", 1 // 1))) - 81.8) < 0.05          # not 91

        # "EVPXI" of a scenario = EVPPI of the partition {scenario}, {the rest}.
        severe = problem("severe", 1 // 1)
        ours = [Float64(evppi(severe, hypothesis_partition(severe, s))) for s in states]
        @test maximum(abs, ours .- nb["evpxi_severe_gamma1"]) <= tol
        # Table 5 of the article is something else (see the fixture).
        @test maximum(abs, ours .- fix["table5"]["evpxi"]) > 8
        @test ours[17] < 2.3 && fix["table5"]["evpxi"][17] == 10.63
        # Equation 6 as printed, with the minimum inside the sum, is just EVPI.
        literal = function (y)
            p = severe.belief.probabilities
            Q = severe.payoff.Q
            rest = setdiff(1:27, y)
            informed = p[y] * maximum(Q[:, y]) + sum(p[s] * maximum(Q[:, s]) for s in rest)
            return informed - act_now_value(severe)
        end
        @test all(literal(y) == evpi(severe) for y in 1:27)

        # "Per-parameter EVPI" of the article: a three-state problem on collapsed
        # shortfalls. Reproduced from the notebook's own collapsed inputs ...
        par = nb["parameter"]
        marginal = rational_vector(par["belief"]["R0"])
        for (name, curve) in par["evpi"]
            x = exact_vector(par["collapsed_shortfall"][name])
            used = rational_vector(par["belief"][name])
            ours3 = [Float64(evpi(DecisionProblem(PayoffTable(actions, [:Low, :Medium, :High],
                                                              table(x, g)), used)))
                     for g in gammas]
            @test maximum(abs, ours3 .- curve) <= 0.006
        end
        cs = exact_vector(par["collapsed_shortfall"]["CS"])
        severe_cs = [Float64(evpi(DecisionProblem(PayoffTable(actions,
                                                              [:Low, :Medium, :High],
                                                              table(cs, g)), marginal)))
                     for g in gammas]
        @test maximum(abs, severe_cs .- par["evpi_CS_under_severe_belief"]) <= 0.006
        # ... which are not the means of Table 4,
        means(axis) = [sum(X[k] for k in 1:27 if triples[k][axis] == l) / 9 for l in 1:3]
        @test round.(Float64.(means(2)); digits=1) == [0.0, 45.3, 870.7]
        @test round.(Float64.(means(1)); digits=1) == [38.3, 348.4, 529.2]
        @test round.(Float64.(means(3)); digits=1) == [220.8, 310.0, 385.2]
        # ... and the collapsed problem is not EVPPI: the loss is not linear in the
        # shortfall, so averaging the shortfall first is not averaging the loss.
        by(axis) = [triples[k][axis] for k in 1:27]
        evppi_cs, evppi_r0, evppi_ci = (evppi(severe, by(2)), evppi(severe, by(1)),
                                        evppi(severe, by(3)))
        collapsed(axis) = evpi(DecisionProblem(PayoffTable(actions, [:Low, :Medium, :High],
                                                           table(means(axis), 1 // 1)),
                                               marginal))
        @test evppi_cs > evppi_r0 > evppi_ci >= 0          # the article's ranking holds
        @test collapsed(2) != evppi_cs && collapsed(1) != evppi_r0
        @test iszero(collapsed(3))                          # the article: zero for CI
        # Partial values are not additive, and none exceeds EVPI.
        @test evppi_cs + evppi_r0 + evppi_ci != evpi(severe)
        @test max(evppi_cs, evppi_r0, evppi_ci) <= evpi(severe)
        both = evppi(severe, [(t[1], t[2]) for t in triples])
        @test max(evppi_cs, evppi_r0) <= both < evppi_cs + evppi_r0
        # The article: with CS and R0 known, CI would not change the action. On Table 4
        # it is worth 16 beds then, and nothing once R0 alone is known.
        @test 15.9 < Float64(evpi(severe) - both) < 16.1
        @test evppi(severe, [(t[1], t[3]) for t in triples]) == evppi_r0
        @test collapsed(2) > evpi(severe)                  # a collapsed value above EVPI
        @test first_choice(decide(severe, MaxExpectedUtility())) == :add_150
        # EVPPI by a parameter is the EVSI of the experiment that reveals it.
        reveal = Experiment(BigRat[by(2)[k] == l ? 1 : 0 for l in 1:3, k in 1:27];
                            signals=[:Low, :Medium, :High], states)
        @test evsi(severe, reveal) == evppi_cs
    end

    @testset "Luhede et al. 2024, water-quality monitoring" begin
        fix = replication("luhede2024_weser")
        actions, states = symbols(fix["actions"]), symbols(fix["states"])
        Q = permutedims(reduce(hcat,
                               [exact_vector(fix["table2"]["payoff"][a])
                                for a in fix["actions"]]))
        prior = rational_vector(fix["prior"])
        problem = DecisionProblem(PayoffTable(actions, states, Q), FiniteBelief(states, prior);
                                  baseline=:a0)
        @test first_choice(decide(problem, MaxExpectedUtility())) == :a1
        @test act_now_value(problem) == target(fix, "prior_value")["value"]
        @test act_now_value(problem) + evpi(problem) == target(fix, "perfect_value")["value"]
        @test evpi(problem) == target(fix, "evpi")["value"]
        value = exact_number(fix["table2"]["good_state_value"])
        general(p, c) = DecisionProblem(PayoffTable(actions, states,
                                                    [0 value; value-c value-c]),
                                        FiniteBelief(states, [1 - p, p]))
        @test general(1 // 10, 90).payoff.Q == Q
        @test evpi(general(1 // 2, 100)) == target(fix, "evpi_cost100_prior_half")["value"]
        grid = (0:100) .// 100
        @test grid[argmax([evpi(general(p, 50)) for p in grid])] ==
              rationalize(target(fix, "evpi_peak_prior_cost50")["value"])
        @test iszero(evpi(general(0 // 1, 90))) && iszero(evpi(general(1 // 1, 90)))
        # The published value of imperfect monitoring, 2.21, cannot be recomputed (the
        # likelihood is not published). It is consistent with the bound, and a binary
        # classifier of the state with equal error rates reaches it at an accuracy of
        # about 0.94 (pinned here; reported in the vignette as ours, not the paper's).
        published = fix["not_replicated"]["imperfect_information_value"]
        @test 0 < published < evpi(problem)
        binary(q) = Experiment([q 1-q; 1-q q]; signals=[:reads_bad, :reads_good], states)
        @test iszero(evsi(problem, binary(1 // 2)))
        @test evsi(problem, binary(1 // 1)) == evpi(problem)
        @test evsi(problem, binary(93 // 100)) < published < evsi(problem, binary(94 // 100))
        # Monitoring is worthless until it can change the decision.
        @test !information_can_change_decision(problem, binary(9 // 10))
        @test information_can_change_decision(problem, binary(19 // 20))
    end

    @testset "Li et al. 2021, musk thistle" begin
        fix = replication("li2021_thistle")
        models = fix["states"]
        states = symbols(models)
        t1 = fix["table1"]
        lambda1 = reduce(hcat, [exact_vector(t1["lambda"][m]) for m in models])
        actions = symbols(t1["actions"])
        problem = DecisionProblem(PayoffTable(actions, states, -lambda1),
                                  FiniteBelief(states, fill(1 // 3, 3));
                                  baseline=:no_management)
        @test first_choice(decide(problem, MaxExpectedUtility())) == :U_solstitialis
        shown = Float64.(exact_vector(t1["printed_average"]))
        @test maximum(abs, Float64.(lambda1 * fill(1 // 3, 3)) .- shown) <= 0.0101
        @test agrees(-act_now_value(problem), fix, "table1_minimum_of_average")
        @test agrees(-act_now_value(problem) - evpi(problem), fix, "table1_average_of_lowest")
        @test agrees(evpi(problem), fix, "table1_evpi")
        @test agrees(100 * evpi(problem) / -act_now_value(problem), fix,
                     "table1_improvement_percent")

        # The matrix models, and the fecundity part reconstructed from the vital rates.
        A = Dict(m => Float64.(reduce(hcat, fix["matrices"]["A"][m])') for m in models)
        function fecundity(m)
            v = fix["vital_rates"]["values"][m]
            F = zeros(4, 4)
            for j in 1:3
                seeds = v["sigma"][j + 1] * v["beta"][j] * v["pi"][j] * v["phi"]
                stay = 1 - v["gamma3"] - v["gamma4"]
                F[:, j + 1] = seeds .* [v["nu"], v["epsilon"] * stay,
                                        v["epsilon"] * v["gamma3"],
                                        v["epsilon"] * v["gamma4"]]
            end
            return F
        end
        F = Dict(m => fecundity(m) for m in models)
        growth(m, f) = maximum(real, eigvals(A[m] .- f .* F[m]))
        @test [round(growth(m, 0); digits=2) for m in models] == Float64.(lambda1[1, :])
        eff = fix["effectiveness"]
        agents = ["R_conicus", "T_horridus", "U_solstitialis"]
        worst = 0.0
        for a in agents, (f, published) in zip(eff["reduction"][a], eff["lambda_check"][a]),
            (k, m) in enumerate(models)

            worst = max(worst, abs(growth(m, f) - published[k]))
        end
        @test worst < 6e-4          # all 65 records x 3 models of Table S8
        # Table 1 again from the seven-digit lambdas: the printed 14.9% is bracketed.
        fine = hcat([growth(m, 0) for m in models],
                    [growth(m, 0.3) for m in models],
                    Float64.(only(eff["other_records"])["lambda"]),
                    [growth(m, 0.7) for m in models])
        fineproblem = DecisionProblem(PayoffTable(actions, states, -permutedims(fine)),
                                      FiniteBelief(states, fill(1 / 3, 3)))
        percent = 100 * evpi(fineproblem) / -act_now_value(fineproblem)
        @test 14.9 < percent < 15.1
        @test 14.7 < 100 * Float64(evpi(problem) / -act_now_value(problem)) < 14.9

        # The joint analysis: three models x 27 combinations of effectiveness quartiles.
        q = Dict(a => quantile(Float64.(eff["reduction"][a]), [0.25, 0.5, 0.75])
                 for a in agents)
        combos = [(m, r, t, u) for m in models for r in 1:3 for t in 1:3 for u in 1:3]
        Q = [-growth(c[1], q[agents[a]][c[a + 1]]) for a in 1:3, c in combos]
        labels = [Symbol(c[1], "_R", c[2], "T", c[3], "U", c[4]) for c in combos]
        joint = DecisionProblem(PayoffTable(Symbol.(agents), labels, Q),
                                FiniteBelief(labels, fill(1 / 81, 81)))
        by_model = [c[1] for c in combos]
        by_effect = [c[2:4] for c in combos]
        base = -act_now_value(joint)
        total, biological, operational = (100 * evpi(joint) / base,
                                          100 * evppi(joint, by_model) / base,
                                          100 * evppi(joint, by_effect) / base)
        published = fix["published_joint"]
        # The best agent in each cell: Australia and USA as in Table 2, New Zealand not.
        letter = Dict(1 => 'R', 2 => 'T', 3 => 'U')
        pattern(m) = String([letter[argmax(Q[:, k])] for k in 1:81 if combos[k][1] == m])
        @test pattern("Australia") == published["table2_pattern"]["Australia"]
        @test pattern("USA") == published["table2_pattern"]["USA"]
        differ = count(collect(pattern("New_Zealand")) .!=
                       collect(published["table2_pattern"]["New_Zealand"]))
        @test differ == published["new_zealand_cells_that_differ"]
        @test pattern("New_Zealand") == pattern("Australia")
        # Our reconstruction, pinned: it does not reproduce the published percentages.
        @test first_choice(decide(joint, MaxExpectedUtility())) == :R_conicus
        @test abs(total - 8.0) < 0.05 && abs(operational - 8.0) < 0.05
        @test abs(biological) < 1e-9
        @test (published["evpi_percent"], published["biological_percent"],
               published["operational_percent"]) == (15.6, 8.5, 10.5)
        @test published["status"] == "not reproduced"
        # What holds in both: partial values are not additive. Published: 8.5 + 10.5 is
        # not 15.6. Ours: the three agents' effectiveness, learned one at a time.
        @test published["biological_percent"] + published["operational_percent"] >
              published["evpi_percent"]
        single = [evppi(joint, [c[a + 1] for c in combos]) for a in 1:3]
        @test all(>=(0), single)
        @test !isapprox(sum(single), evppi(joint, by_effect); rtol=1e-3)
        @test maximum(single) <= evppi(joint, by_effect) + 1e-12 <= evpi(joint) + 2e-12
    end

    @testset "the functions added for the replications" begin
        p = DecisionProblem(PayoffTable([4 0 0; 1 2 3]), [1 // 2, 1 // 4, 1 // 4])
        @test hypothesis_partition(p, :s2) == [:other, :s2, :other]
        @test hypothesis_partition(["a", "b"], "a") == [:a, :other]
        @test_throws ArgumentError hypothesis_partition(p, :nope)
        @test_throws ArgumentError hypothesis_partition([:other, :x], :other)
        # two_stage_value: one first stage for all actions is perfect information;
        # every action its own first stage is acting now.
        for seed in 1:20
            rng = Random.Xoshiro(seed)
            Q = rand(rng, -9:9, 6, 4) .// 1
            w = rand(rng, 1:5, 4)
            prob = DecisionProblem(PayoffTable(Q), w .// sum(w))
            @test two_stage_value(prob, fill(:g, 6)).value ==
                  act_now_value(prob) + evpi(prob)
            alone = two_stage_value(prob, 1:6)
            @test alone.value == alone.act_now == act_now_value(prob) && iszero(alone.gain)
            part = [1, 1, 2, 2]
            r = two_stage_value(prob, [1, 1, 2, 2, 3, 3]; partition=part)
            @test r.act_now <= r.value <= r.act_now + evppi(prob, part)
            @test two_stage_value(prob, [1, 1, 2, 2, 3, 3]; partition=fill(0, 4)).gain == 0
            @test r.value == maximum(r.values) && r.chosen == r.first_stages[r.values .== r.value]
        end
        masked = DecisionProblem(PayoffTable([:a, :b, :c], [:s, :t], [9 9; 1 0; 0 1]),
                                 [1 // 2, 1 // 2]; feasible=[:b, :c])
        @test two_stage_value(masked, [:x, :y, :y]).first_stages == [:y]
        @test two_stage_value(masked, [:x, :y, :y]).value == 1
        @test_throws ArgumentError two_stage_value(p, [:a])
        @test_throws ArgumentError two_stage_value(p, [:a, :b]; partition=[1, 2])
        credal = DecisionProblem(PayoffTable([4 0; 1 2]), CredalSet([[1 // 2, 1 // 2], [1 // 4, 3 // 4]]))
        @test_throws BeliefFormError two_stage_value(credal, [:a, :b])
        # gaussian_experiment: a proper experiment, nearly perfect when the error is
        # small and nearly useless when it is large; argument checks.
        e = gaussian_experiment([0.0, 1.0, 1.0], 0.05; bins=600)
        @test size(e.likelihood) == (600, 3)
        @test all(isapprox.(sum(e.likelihood; dims=1), 1; atol=1e-12))
        @test e.likelihood[:, 2] == e.likelihood[:, 3]
        g = DecisionProblem(PayoffTable([1.0 0.0 0.0; 0.0 1.0 2.0]), [0.5, 0.25, 0.25])
        @test isapprox(evsi(g, e), evppi(g, [1, 2, 2]); atol=1e-9)
        @test evsi(g, gaussian_experiment([0.0, 1.0, 1.0], 50.0)) < 1e-3
        # Closed form for two states, equal priors, 0-1 payoff: Phi(d / 2 sd) - 1/2.
        two = DecisionProblem(PayoffTable([1.0 0.0; 0.0 1.0]), [0.5, 0.5])
        @test isapprox(evsi(two, gaussian_experiment([0.0, 1.0], 0.5; bins=4000)),
                       0.3413447460685429; atol=1e-6)
        @test_throws ArgumentError gaussian_experiment([0.0, 1.0], 0.0)
        @test_throws ArgumentError gaussian_experiment(Float64[], 1.0)
        @test_throws ArgumentError gaussian_experiment([0.0, 1.0], 1.0; bins=1)
        @test_throws ArgumentError gaussian_experiment([0.0, Inf], 1.0)
    end
end
