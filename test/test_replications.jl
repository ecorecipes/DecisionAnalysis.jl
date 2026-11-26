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

# The model of Bal et al. 2018 (fixture bal2018_monitoring): 64 models of which threat
# causes which species to decline, a budget split between two threats, and observations
# of declines with and without an experimental removal of one threat.
const BAL_MODELS = vec([permutedims(reshape(collect(d), 3, 2))
                        for d in Iterators.product(ntuple(_ -> (0, 1), 6)...)])
const BAL_STATES = [Symbol("model", k) for k in 1:64]
function bal_declines(d, q, c, m1, m2; wallaby=false)
    r1, r2 = (c[1] - m1) / c[1], (c[2] - m2) / c[2]
    return sum(1:3) do j
        second = wallaby && j == 3 ? d[2, j] - d[2, j] * r2 : 1 - d[2, j] * r2
        return q[j] + (1 - q[j]) * (1 - (1 - d[1, j] * r1) * second)
    end
end
# Utility is minus the number of species declining. The first action spends nothing
# (the baseline); the others spend the whole budget, on a grid of splits.
function bal_problem(p, q, c, budget; wallaby=false, n=2000)
    lo, hi = max(0.0, budget - c[2]), min(budget, c[1])
    grid = [lo + max(hi - lo, 0.0) * g / n for g in 0:n]
    Q = zeros(n + 2, 64)
    for (s, d) in enumerate(BAL_MODELS)
        Q[1, s] = -bal_declines(d, q, c, 0.0, 0.0; wallaby)
        for (g, m1) in enumerate(grid)
            Q[g + 1, s] = -bal_declines(d, q, c, m1, budget - m1; wallaby)
        end
    end
    prior = [prod(d[i, j] == 1 ? p[i, j] : 1 - p[i, j] for i in 1:2, j in 1:3)
             for d in BAL_MODELS]
    actions = vcat(:no_management, [Symbol("split", g) for g in 0:n])
    return DecisionProblem(PayoffTable(actions, BAL_STATES, Q),
                           FiniteBelief(BAL_STATES, prior ./ sum(prior));
                           baseline=:no_management)
end
function bal_observation(d, q, j, alpha, beta, manage)
    cause = maximum(i == manage ? 0 : d[i, j] for i in 1:2)
    D = q[j] + (1 - q[j]) * cause
    return D * (1 - beta) + (1 - D) * alpha
end
function bal_experiment(q, species; manage=nothing, alpha=0.0, beta=0.0)
    probs = Vector{Float64}[]
    for j in species
        push!(probs, [bal_observation(d, q, j, alpha, beta, nothing) for d in BAL_MODELS])
        manage === nothing ||
            push!(probs,
                  [bal_observation(d, q, j, alpha, beta, manage) for d in BAL_MODELS])
    end
    outcomes = vec(collect(Iterators.product(ntuple(_ -> (0, 1), length(probs))...)))
    L = [prod(x[k] == 1 ? probs[k][s] : 1 - probs[k][s] for k in eachindex(probs))
         for x in outcomes, s in 1:64]
    return Experiment(L; signals=[Symbol("x", join(x)) for x in outcomes],
                      states=BAL_STATES)
end

# The epidemic of Atkins et al. 2020 (fixture atkins2020_vaccination): a deterministic
# SEIR model in which susceptibles are vaccinated at a fixed daily rate while a campaign
# runs and the pool lasts. The state is (S, E, I, R, new infections, vaccines given).
function atkins_rates(u, p, vaccinating)
    S, E, I = u[1], u[2], u[3]
    given = vaccinating && u[6] < p.pool ? min(p.per_day, max(S, 0.0)) : 0.0
    infection = p.beta * S * I / p.N
    return (-infection - p.efficacy * given, infection - p.sigma * E,
            p.sigma * E - p.gamma * I, p.gamma * I + p.efficacy * given, infection, given)
end
function atkins_step(u, p, vaccinating, dt)
    k1 = atkins_rates(u, p, vaccinating)
    k2 = atkins_rates(u .+ dt / 2 .* k1, p, vaccinating)
    k3 = atkins_rates(u .+ dt / 2 .* k2, p, vaccinating)
    k4 = atkins_rates(u .+ dt .* k3, p, vaccinating)
    return u .+ dt / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end
# Duration, new infections and vaccines given from the day of detection (20 infected)
# for a campaign that runs from day `from` to day `to`.
function atkins_campaign(p, from, to; dt=0.01)
    u = (5000.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    while u[3] < 20
        u = atkins_step(u, p, false, dt)
    end
    u, t = (u[1], u[2], u[3], u[4], 0.0, 0.0), 0.0
    while u[2] + u[3] >= 1
        u = atkins_step(u, p, from <= t < to, dt)
        t += dt
    end
    return (; days=t, infections=u[5], vaccines=min(u[6], p.pool))
end
# The cost of each of the four campaigns at each efficacy, and the two-period problem:
# a state is an efficacy together with what was done in the first week, the first
# period pays nothing and moves the state, the second pays minus the cost of the whole
# campaign, and vaccinating shows the successes among the monitored vaccinations.
function atkins_problem(s; monitored=35, bins=100, decision_day=7.0)
    number(key) = Float64(exact_number(s[key]))
    efficacies = [(k - 0.5) / bins for k in 1:bins]
    prior = [x^0.1 * (1 - x)^0.1 for x in efficacies]
    prior ./= sum(prior)
    campaigns = (vaccinate_vaccinate=(0.0, Inf), vaccinate_not=(0.0, decision_day),
                 not_vaccinate=(decision_day, Inf), not_not=(0.0, 0.0))
    cost = map(campaigns) do (from, to)
        return map(efficacies) do efficacy
            p = (; beta=number("beta"), sigma=number("sigma"), gamma=number("gamma"),
                 N=5001.0, efficacy, per_day=100.0, pool=number("pool"))
            run = atkins_campaign(p, from, to)
            return number("weight_duration") * run.days +
                   number("weight_infection") * run.infections +
                   number("weight_vaccine") * run.vaccines +
                   number("weight_campaign") * (to > from)
        end
    end
    n = bins
    states = vcat([Symbol(phase, k) for phase in (:start, :vaccinated, :waited)
                   for k in 1:n])
    Q = zeros(2, 3n)
    Q[1, (n + 1):(2n)], Q[2, (n + 1):(2n)] = -cost.vaccinate_vaccinate, -cost.vaccinate_not
    Q[1, (2n + 1):(3n)], Q[2, (2n + 1):(3n)] = -cost.not_vaccinate, -cost.not_not
    vaccinating, waiting = zeros(3n, 3n), zeros(3n, 3n)
    for k in 1:n
        vaccinating[n + k, k] = waiting[2n + k, k] = 1
        for later in (n + k, 2n + k)
            vaccinating[later, later] = waiting[later, later] = 1
        end
    end
    likelihood = [binomial(monitored, x) * e^x * (1 - e)^(monitored - x)
                  for x in 0:monitored, e in repeat(efficacies, 3)]
    seen = (;
            vaccinate=Experiment(likelihood;
                                 signals=[Symbol("successes", x)
                                          for x in 0:monitored], states))
    problem = DecisionProblem(PayoffTable([:vaccinate, :not], states, Q),
                              vcat(prior, zeros(2n)); baseline=:not)
    return (; problem, seen, moves=(; vaccinate=vaccinating, not=waiting), cost, prior,
            efficacies)
end
function atkins_solve(m)
    return adaptive_management(m.problem, m.seen; horizon=2, transitions=m.moves,
                               depth=2)
end
# The fewest successes after which the active manager continues vaccinating.
function atkins_successes_needed(result)
    continued = [parse(Int, String(z)[10:end])
                 for (z, node) in result.policy.branches
                 if node.action == :vaccinate]
    stopped = [parse(Int, String(z)[10:end])
               for (z, node) in result.policy.branches
               if node.action == :not]
    @assert maximum(stopped) < minimum(continued)
    return minimum(continued)
end

# The pooled projections of Shea et al. 2023 (fixture shea2023_covid_models). A
# submission is a list of values at consecutive quantile levels (per cent); its
# distribution function is taken as linear between them.
function shea_cdf(first_level, values, x)
    x < values[1] && return 0.0
    x >= values[end] && return 1.0
    j = searchsortedlast(values, x)
    lower, upper = values[j], values[j + 1]
    p = (first_level + j - 1) / 100
    return upper > lower ? p + (x - lower) / (upper - lower) / 100 : p + 1 / 100
end
# The mean of a submission: the area under its quantile function.
function shea_mean(values)
    length(values) == 1 && return values[1]
    return sum((values[k] + values[k + 1]) / 2 for k in 1:(length(values) - 1)) /
           (length(values) - 1)
end
@testset "replications of published analyses" begin
    @testset "every fixture states its provenance" begin
        files = filter(endswith(".json"), readdir(REPLICATIONS))
        @test length(files) == 14
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
                    return w[k] * maximum(-S1[r, k]
                                          for r in eachindex(courses)
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
        fp = DecisionProblem(PayoffTable(actions, states, Q),
                             FiniteBelief(states, Float64.(w)))
        total = evpi(fp)
        fraction(a, sd; bins=1200) = evsi(fp,
                                          gaussian_experiment(Q[findfirst(==(a), actions),
                                                                :],
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
        states = [Symbol(levels[r], "_", levels[c], "_", levels[i])
                  for (r, c, i) in triples]
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
        @test all(iszero(evpi(problem(b, 0 // 1)))
                  for b in ("none", "low", "medium", "severe"))
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
                  text["evpi_none_gamma08"]["value"]) <=
              text["evpi_none_gamma08"]["tolerance"]
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
            ours3 = [Float64(evpi(DecisionProblem(PayoffTable(actions,
                                                              [:Low, :Medium, :High],
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
        problem = DecisionProblem(PayoffTable(actions, states, Q),
                                  FiniteBelief(states, prior);
                                  baseline=:a0)
        @test first_choice(decide(problem, MaxExpectedUtility())) == :a1
        @test act_now_value(problem) == target(fix, "prior_value")["value"]
        @test act_now_value(problem) + evpi(problem) ==
              target(fix, "perfect_value")["value"]
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
        @test evsi(problem, binary(93 // 100)) < published <
              evsi(problem, binary(94 // 100))
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
        @test agrees(-act_now_value(problem) - evpi(problem), fix,
                     "table1_average_of_lowest")
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

    @testset "Kim et al. 2017, malaria vector control" begin
        fix = replication("kim2017_malaria")
        actions = symbols(fix["actions"])
        function kim_problem(case; weights=nothing)
            c = fix["table4"][case]
            function marginal(name)
                return weights === nothing ?
                       exact_vector(fix["table3"][name]["probabilities"]) : weights
            end
            first, second = marginal(c["first"]), marginal(c["second"])
            prior = BigRat[a * b for a in first for b in second]
            states = [Symbol(case, "_", i, j) for i in 1:3 for j in 1:3]
            Q = vcat(zeros(BigRat, 1, 9),
                     reduce(hcat, [exact_vector(r) for r in c["payoff"]]))
            return DecisionProblem(PayoffTable(actions, states, Q),
                                   FiniteBelief(states, prior); baseline=:status_quo),
                   states
        end
        for case in ("A", "B", "C", "D")
            problem, states = kim_problem(case)
            @test first_choice(decide(problem, MaxExpectedUtility())) == :larviciding_8
            @test agrees(act_now_value(problem), fix, case * "_prior_value")
            @test agrees(act_now_value(problem) + evpi(problem), fix,
                         case * "_perfect_value")
            @test agrees(evpi(problem), fix, case * "_evpi")
            # Ours, not the paper's: what each parameter of the pair is worth alone.
            one = evppi(problem, [div(k - 1, 3) for k in 1:9])
            other = evppi(problem, [mod1(k, 3) for k in 1:9])
            @test 0 <= one <= evpi(problem) && 0 <= other <= evpi(problem)
        end
        # The printed EVPI is the difference of the rounded values (fixture, resolution).
        @test round(Float64(evpi(first(kim_problem("A")))); digits=2) == 0.38
        @test round(Float64(evpi(first(kim_problem("B")))); digits=2) == 4.17
        # The saturation the paper reports: the two higher values of the fitness cost
        # (and of the evolutionary speed) give the same payoffs, so in case D only the
        # low evolutionary speed is worth telling apart from the rest.
        d, states = kim_problem("D")
        @test d.payoff.Q[:, 4:6] == d.payoff.Q[:, 7:9]
        @test evppi(d, [k <= 3 for k in 1:9]) == evppi(d, [div(k - 1, 3) for k in 1:9])
        # The uniform column cannot be recomputed: equal thirds do not give it, and in
        # case D they make another action the best one without information.
        third = fill(big(1) // 3, 3)
        uniform_a = first(kim_problem("A"; weights=third))
        @test abs(Float64(act_now_value(uniform_a)) -
                  fix["not_replicated"]["uniform"]["A"][1]) > 0.5
        @test first_choice(decide(first(kim_problem("D"; weights=third)),
                                  MaxExpectedUtility())) == :irs_larviciding_4
    end

    @testset "Christensen et al. 2026, alligator snapping turtle" begin
        fix = replication("christensen2026_turtle")
        actions, states = symbols(fix["actions"]), symbols(fix["states"])
        prior = exact_vector(fix["prior"])
        @test sum(prior) == 1
        table(rows) = permutedims(reduce(hcat, [exact_vector(r) for r in rows]))
        turtle(Q) = DecisionProblem(PayoffTable(actions, states, permutedims(Q)),
                                    FiniteBelief(states, prior); baseline=:no_action)
        curves = ("Convex" => "convex", "Linear" => "linear", "S-curve" => "s_curve",
                  "Concave" => "concave")
        best = Dict("lambda" => :reduce_extent_bycatch,
                    "time_to_extirpation" => :reduce_extent_bycatch,
                    "fecundity" => :reduce_extent_nest_predators)
        # From the printed payoff tables: all of Table 3.
        for metric in ("lambda", "time_to_extirpation", "fecundity"), (block, id) in curves
            problem = turtle(table(fix["payoffs"][metric][block]))
            @test best[metric] in decide(problem, MaxExpectedUtility()).chosen
            @test agrees(act_now_value(problem), fix, "$(metric)_$(id)_uncertainty")
            @test agrees(act_now_value(problem) + evpi(problem), fix,
                         "$(metric)_$(id)_certainty")
            @test agrees(evpi(problem), fix, "$(metric)_$(id)_evpi")
        end
        # The largest value in the paper, unrounded (fixture, resolution).
        headline = turtle(table(fix["payoffs"]["time_to_extirpation"]["S-curve"]))
        @test round(Float64(evpi(headline)); digits=5) == 0.01434
        @test round(100 * Float64(evpi(headline) / act_now_value(headline)); digits=2) ==
              5.94
        # From the raw model outputs through the value functions as printed.
        logistic(f) = x -> Float64(exact_number(f["scale"])) /
                           (1 + exp(-12 * (x - Float64(exact_number(f["midpoint"]))))) +
                           Float64(exact_number(f["shift"]))
        functions = fix["value_functions"]
        raw = Dict("lambda" => "lambda", "time_to_extirpation" => "years",
                   "fecundity" => "fecundity")
        scaled(metric) = Float64.(table(fix["raw"][raw[metric]])) ./
                         Float64(exact_number(fix["scaling"][raw[metric]]))
        deviation(metric, block, f) = begin
            ours, printed = f.(scaled(metric)),
                            Float64.(table(fix["payoffs"][metric][block]))
            # The no-action column of the time table is printed as zero (resolution).
            keep = metric == "time_to_extirpation" ? (1:5) : (1:6)
            maximum(abs, ours[:, keep] .- printed[:, keep])
        end
        for metric in keys(raw)
            @test deviation(metric, "Linear", identity) < 1e-12
            @test deviation(metric, "Convex", logistic(functions["convex"])) < 6e-5
            @test deviation(metric, "S-curve", logistic(functions["s_curve"])) < 6e-5
            @test deviation(metric, "Concave",
                            logistic(functions["concave_fitted_by_us"])) <
                  6e-5
        end
        # The concave function as printed does not give the printed concave table.
        @test deviation("lambda", "Concave", logistic(functions["concave_as_printed"])) >
              0.15
        printed_concave = turtle(logistic(functions["concave_as_printed"]).(scaled("lambda")))
        @test round(Float64(act_now_value(printed_concave)); digits=4) == 0.4125
        # The raw years, before any value function (Table S5).
        years = turtle(table(fix["raw"]["years"]))
        @test Float64.(aggregate(Expectation(), years)) ≈
              [9.34, 6.785, 0.055, 6.06, 0.345, 0]
        @test expected_gain_over_baseline(years)[1] == act_now_value(years)
        # Ours, not the paper's: the two halves of the uncertainty, taken alone.
        magnitude = [div(k - 1, 3) for k in 1:9]
        extent = [mod1(k, 3) for k in 1:9]
        @test 0 <= evppi(headline, magnitude) <= evpi(headline)
        @test 0 <= evppi(headline, extent) <= evpi(headline)
    end

    @testset "Bal et al. 2018, monitoring species and threats" begin
        fix = replication("bal2018_monitoring")
        s2 = fix["table_s2"]
        number(x) = Float64(exact_number(x))
        p, q = fill(number(s2["p"]), 2, 3), fill(number(s2["q"]), 3)
        c, budget = number.(s2["c"]), number(s2["budget"])
        singles, pairs = [[1], [2], [3]], [[1, 2], [1, 3], [2, 3]]
        problem = bal_problem(p, q, c, budget)
        @test length(problem.payoff.states) == 64
        @test abs(evpi(problem) - s2["evpi"]) <= s2["tolerance"]
        for row in s2["rows"]
            alpha, beta = number(row["alpha"]), number(row["beta"])
            for species in vcat(singles, pairs)
                # Watching without intervening tells nothing here.
                @test abs(evsi(problem, bal_experiment(q, species; alpha, beta))) < 1e-12
                printed = row[length(species) == 1 ? "one_species" : "two_species"]
                for threat in 1:2
                    e = bal_experiment(q, species; manage=threat, alpha, beta)
                    @test abs(evsi(problem, e) - printed[threat]) <= s2["tolerance"]
                    @test evsi(problem, e) <= evpi(problem)
                end
            end
        end
        # No value at an empty or a full budget.
        @test abs(evpi(bal_problem(p, q, c, 0.0))) < 1e-12
        @test abs(evpi(bal_problem(p, q, c, 1.0))) < 1e-12

        case(name) = begin
            t = fix["table_s1"][name]
            (p=permutedims(reduce(hcat, [number.(r) for r in t["p"]])), q=number.(t["q"]),
             c=number.(t["c"]))
        end
        strategies = [(species, threat) for threat in 1:2
                      for species in vcat(singles, pairs)]
        function sweep(name; wallaby)
            k = case(name)
            map(0.1:0.1:0.9) do share
                pr = bal_problem(k.p, k.q, k.c, share * sum(k.c); wallaby)
                return (; evpi=evpi(pr),
                        evsi=Dict(s => evsi(pr, bal_experiment(k.q, s[1]; manage=s[2]))
                                  for s in strategies),
                        surveillance=maximum(evsi(pr, bal_experiment(k.q, s))
                                             for s in vcat(singles, pairs)))
            end
        end
        best(row, n) = argmax(s -> row.evsi[s], filter(s -> length(s[1]) == n, strategies))

        fitz = sweep("fitz_stirling"; wallaby=true)
        at30 = fitz[3]
        @test agrees(at30.evpi, fix, "fitz_evpi")
        @test argmax(i -> fitz[i].evpi, 1:9) == 3
        @test agrees(at30.evsi[best(at30, 2)], fix, "fitz_best_evsi")
        @test agrees(at30.evsi[best(at30, 2)] / at30.evsi[best(at30, 1)], fix,
                     "fitz_two_over_one")
        # The bandicoot and the possum, with fire managed experimentally, at every budget.
        @test all(row -> best(row, 2) == ([1, 2], 2), fitz)
        # One species: the possum at the low and the middle-high budgets, the bandicoot
        # between, always with fire (Section 3.2).
        @test [best(row, 1) for row in fitz[1:6]] ==
              [([2], 2), ([1], 2), ([1], 2), ([1], 2), ([2], 2), ([2], 2)]
        # Without the wallaby term of equation S4.1 the published numbers are not reached.
        standard = sweep("fitz_stirling"; wallaby=false)[3]
        @test round(standard.evpi; digits=4) == 0.0504
        @test round(standard.evsi[best(standard, 2)]; digits=4) == 0.0153

        pilbara = sweep("pilbara"; wallaby=false)
        top = argmax(i -> pilbara[i].evsi[best(pilbara[i], 2)], 1:9)
        @test top == 6
        @test agrees(pilbara[top].evsi[best(pilbara[top], 2)], fix, "pilbara_best_evsi")
        @test agrees(maximum(row.evsi[best(row, 2)] / row.evpi for row in pilbara), fix,
                     "pilbara_share_of_evpi")
        @test all(row -> row.surveillance < 1e-9, pilbara)
        # The bilby is the species to monitor at every budget; the experiment changes
        # from grazing to fire at 60% (Results; the Discussion says otherwise).
        @test all(row -> best(row, 1)[1] == [1], pilbara)
        @test [best(row, 1)[2] for row in pilbara[1:6]] == [1, 1, 1, 1, 1, 2]
        @test all(row -> best(row, 2) == ([1, 2], 1), pilbara)
    end

    @testset "Chadès et al. 2008, a cryptic tiger population" begin
        fix = replication("chades2008_tiger")
        actions, states = symbols(fix["actions"]), symbols(fix["states"])
        @test contains("signal_of = :next")(fix["model"]["observation"])
        # The model from one set of printed values; `detection` replaces the detection
        # probability of a survey.
        function tiger(from; detection=nothing, T=BigRat)
            number(key) = T(exact_number(from[key]))
            V, cm, cs = number("value"), number("cost_manage"), number("cost_survey")
            d = detection === nothing ? number("detection_surveyed") : T(detection)
            d0 = number("detection_not_surveyed")
            # Reward: V while extant, less the cost of the action (the fixture's reading).
            table = PayoffTable(actions, states, T[V 0; V-cm -cm; V-cs -cs])
            dies(q) = T[1-q 0; q 1]                       # extinction is absorbing
            moves = (; nothing=dies(number("extinction_unmanaged")),
                     manage=dies(number("extinction_managed")),
                     survey=dies(number("extinction_unmanaged")))
            looks(q) = Experiment(T[1-q 1; q 0]; signals=symbols(fix["signals"]), states)
            seen = (; nothing=looks(d0), manage=looks(d0), survey=looks(d))
            return table, moves, seen
        end
        function solved(table, moves, seen; horizon, kwargs...)
            T = eltype(table.Q)
            return adaptive_management(DecisionProblem(table, T[1, 0]; baseline=:nothing),
                                       seen; horizon, transitions=moves, signal_of=:next,
                                       depth=0, kwargs...)
        end
        # What is done year by year after a sighting with `years` to go, if the species
        # is never seen again, and the belief before each decision.
        function unobserved(result, moves, seen, years)
            T = eltype(first(moves))
            belief, path, beliefs = T[1, 0], Symbol[], T[]
            for n in years:-1:1
                pieces = result.value_function[n]
                a = pieces.actions[argmax(pieces.vectors * belief)]
                push!(path, a)
                push!(beliefs, belief[1])
                joint = (moves[a] * belief) .* seen[a].likelihood[1, :]
                belief = joint ./ sum(joint)
            end
            return path, beliefs
        end
        spell(path, a) = count(==(a), path)
        # The belief at which surveying gives way to managing with n years to go: where
        # the highest survey line meets the highest manage line.
        function switch_to_manage(result, n)
            pieces = result.value_function[n]
            G = pieces.vectors
            order = sortperm([G[i, 1] - G[i, 2] for i in axes(G, 1)])
            i, j = only((i, j)
                        for (i, j) in zip(order[1:(end - 1)], order[2:end])
                        if pieces.actions[i] == :survey && pieces.actions[j] == :manage)
            return (G[j, 2] - G[i, 2]) / ((G[i, 1] - G[i, 2]) - (G[j, 1] - G[j, 2]))
        end

        table, moves, seen = tiger(fix["table_s1"])
        long = solved(table, moves, seen; horizon=100, method=:vectors, passive=false)
        @test long.passive === nothing && long.value_of_probing === nothing
        # The value function is a handful of lines, where the beliefs are beyond count.
        @test all(size(pieces.vectors, 1) <= 20 for pieces in long.value_function)
        path, beliefs = unobserved(long, moves, seen, 100)
        # The published schedule: manage, then survey, then stop.
        @test path == vcat(fill(:manage, 12), fill(:survey, 3), fill(:nothing, 85))
        @test spell(path, :manage) == target(fix, "years_managing")["value"]
        @test spell(path, :survey) == target(fix, "years_surveying")["value"]
        # The value function with n years to go does not depend on the horizon, so the
        # one run gives the schedule for every shorter horizon.
        schedules = map(1:100) do years
            shorter, _ = unobserved(long, moves, seen, years)
            return (spell(shorter, :manage), spell(shorter, :survey))
        end
        ours = fix["ours"]
        pair(d) = (d["managing"], d["surveying"])
        @test schedules[30] == pair(ours["thirty_years"]) == (11, 2)
        @test schedules[60] == pair(ours["sixty_years"])
        @test schedules[100] == pair(ours["hundred_years"])
        @test findfirst(h -> all(==((12, 3)), schedules[h:end]), 1:100) ==
              ours["first_horizon_with_published_schedule"]
        # Why 30 years gives 11: the belief after 11 unobserved years of managing lies
        # between the switching belief with 19 years left and the long-run one.
        thirty, thirty_beliefs = unobserved(long, moves, seen, 30)
        @test thirty[12] == :survey
        @test round(Float64(thirty_beliefs[12]); digits=3) == 0.501
        @test round(Float64(switch_to_manage(long, 19)); digits=3) == 0.519
        @test round(Float64(switch_to_manage(long, 100)); digits=3) == 0.491
        @test switch_to_manage(long, 100) < thirty_beliefs[12] < switch_to_manage(long, 19)
        # SI Appendix: with one year left surveying is never optimal (nor is managing).
        @test long.value_function[1].actions == [:nothing]
        @test all(!(:survey in long.value_function[n].actions) for n in 1:3)

        # The two methods agree exactly where both can run, and the four values are in
        # order.
        short_beliefs = solved(table, moves, seen; horizon=8)
        short_vectors = solved(table, moves, seen; horizon=8, method=:vectors)
        for field in (:no_learning, :passive, :active, :perfect, :baseline, :plan,
                      :first_action)
            @test getfield(short_beliefs, field) == getfield(short_vectors, field)
        end
        full = solved(table, moves, seen; horizon=30, method=:vectors)
        @test full.baseline < full.no_learning <= full.passive < full.active < full.perfect
        @test full.active == long.value_function[30].vectors[1, 1] ||
              full.active == maximum(long.value_function[30].vectors[:, 1])
        # In floating point the schedule is the same.
        ftable, fmoves, fseen = tiger(fix["table_s1"]; T=Float64)
        floating = solved(ftable, fmoves, fseen; horizon=100, method=:vectors,
                          passive=false)
        @test first(unobserved(floating, fmoves, fseen, 100)) == path
        @test floating.active ≈ long.active rtol = 1e-12

        # The rounded values of the main text give the same schedules.
        mtable, mmoves, mseen = tiger(fix["main_text_values"])
        rounded = solved(mtable, mmoves, mseen; horizon=100, method=:vectors,
                         passive=false)
        main100, _ = unobserved(rounded, mmoves, mseen, 100)
        main30, _ = unobserved(rounded, mmoves, mseen, 30)
        @test (spell(main100, :manage), spell(main100, :survey)) == (12, 3)
        @test (spell(main30, :manage), spell(main30, :survey)) == (11, 2)

        # Fig. S2: with perfect detection one survey settles it.
        ptable, pmoves, pseen = tiger(fix["table_s1"]; detection=1)
        perfect = solved(ptable, pmoves, pseen; horizon=100, method=:vectors,
                         passive=false)
        sure, _ = unobserved(perfect, pmoves, pseen, 100)
        @test spell(sure, :survey) ==
              target(fix, "years_surveying_perfect_detection")["value"]
    end

    @testset "Li et al. 2019, Ebola: epidemiological and operational uncertainty" begin
        fix = replication("li2019_ebola")
        actions = symbols(fix["actions"])
        rows = fix["table_s3"]["rows"]
        @test length(rows) == 37 * 27 == 999
        @test length(fix["table_s3"]["columns"]) == 13 == length(first(rows))
        models = [Int(r[1]) for r in rows]
        combinations = [(Int(r[2]), Int(r[3]), Int(r[4])) for r in rows]
        @test sort(unique(models)) == 1:37 && length(unique(combinations)) == 27
        @test allunique(zip(models, combinations))
        states = [Symbol("model", m, "_cost", c...) for (m, c) in zip(models, combinations)]
        # Minus the caseload of Table S3, exactly, for one budget: rows are the three
        # interventions, columns the 999 pairs of a model and a cost combination.
        function ebola(budget)
            first_column = 5 +
                           3 * (findfirst(==(budget), ["low", "intermediate", "high"]) - 1)
            cases = permutedims(reduce(hcat,
                                       [exact_vector(r[first_column:(first_column + 2)])
                                        for r in rows]))
            return DecisionProblem(PayoffTable(actions, states, -permutedims(cases)),
                                   FiniteBelief(states, fill(1 // big(999), 999));
                                   baseline=Symbol(fix["baseline"]))
        end
        # The best intervention for each group of states, on the group's average.
        function best_by(problem, labels)
            Q = problem.payoff.Q
            return Dict(g => actions[argmax([sum(Q[a, j]
                                                 for j in eachindex(labels)
                                                 if labels[j] == g)
                                             for a in eachindex(actions)])]
                        for g in unique(labels))
        end
        for budget in ("low", "intermediate", "high")
            problem = ebola(budget)
            minimum_of_average = -act_now_value(problem)
            # Reducing community transmission is best under present knowledge at
            # every budget, which is why it serves as the baseline.
            @test first_choice(decide(problem, MaxExpectedUtility())) == :community
            @test all(>=(0), expected_gain_over_baseline(problem)[[1]])
            total = evpi(problem)
            by_model, by_cost = evppi(problem, models), evppi(problem, combinations)
            for (id, ours) in (("minimum_of_average", minimum_of_average),
                               ("average_of_lowest", minimum_of_average - total),
                               ("evpi", total),
                               ("evpi_percent", 100 * total / minimum_of_average),
                               ("average_of_lowest_across_models",
                                minimum_of_average - by_model),
                               ("evxi_epidemiological", by_model),
                               ("evxi_epidemiological_percent",
                                100 * by_model / minimum_of_average),
                               ("average_of_lowest_across_cost_functions",
                                minimum_of_average - by_cost),
                               ("evxi_operational", by_cost),
                               ("evxi_operational_percent",
                                100 * by_cost / minimum_of_average))
                @test agrees(ours, fix, id * "_" * budget)
            end
            # Partial values are bounded by the whole and do not add up to it.
            @test 0 < by_cost < total && 0 < by_model < total
            @test by_model + by_cost != total
            # Knowing both is knowing the state.
            @test evppi(problem, collect(zip(models, combinations))) == total
        end
        # The text transposes two percentages that its Table 2 has in this order.
        low, middle = ebola("low"), ebola("intermediate")
        share(p) = round(Float64(100 * evppi(p, combinations) / -act_now_value(p));
                         digits=1)
        @test (share(low), share(middle)) == (11.3, 13.1)
        # With more money the two sources change places.
        high = ebola("high")
        @test evppi(low, combinations) > evppi(low, models)
        @test evppi(high, models) > 10 * evppi(high, combinations)

        # The counts quoted for the low budget (Table 1 of the article).
        over_models = best_by(low, combinations)      # one best action per combination
        over_costs = best_by(low, models)             # one best action per model
        @test count(==(:community), values(over_models)) ==
              target(fix, "community_best_cost_combinations_low")["value"]
        @test count(==(:funeral), values(over_costs)) ==
              target(fix, "funeral_best_models_low")["value"]
        @test first_choice(decide(low, MaxExpectedUtility())) == :community
        # An intervention that is cheap while both others are expensive (cost function
        # 2), counted over the 37 models, in both of its cheap forms (1 and 3).
        function wins(action, combination)
            a = findfirst(==(action), actions)
            return count(j -> combinations[j] == combination &&
                              argmax(view(low.payoff.Q, :, j)) == a, eachindex(states))
        end
        # The count for community transmission is the paper's (37 of 37). Those for the
        # other two are not (36 and 31 in the text): the fixture pins ours, and no
        # single combination of Table S3 gives the published ones.
        cheap_forms = ((1, "cheap_and_effective"), (3, "cheap_and_partly_effective"))
        for (cheap, _) in cheap_forms
            @test wins(:community, ntuple(k -> k == 1 ? cheap : 2, 3)) ==
                  target(fix, "community_best_models_when_cheap_others_expensive")["value"]
        end
        for (action, place) in ((:funeral, 2), (:hospitalization, 3))
            id = String(action) * "_best_models_when_cheap_others_expensive"
            for (cheap, form) in cheap_forms
                ours = wins(action, ntuple(k -> k == place ? cheap : 2, 3))
                @test ours == fix["ours"][id][form]
                @test ours < target(fix, id)["value"]
            end
            published = target(fix, id)["value"]
            @test !any(wins(action, c) == published for c in unique(combinations))
        end
    end

    @testset "Atkins et al. 2020, vaccinating under unknown efficacy" begin
        fix = replication("atkins2020_vaccination")
        choices = fix["published_choices"]
        expected(m, costs) = sum(m.prior .* costs)
        crossover(m) = m.efficacies[findfirst(m.cost.vaccinate_vaccinate .<
                                              m.cost.vaccinate_not)] - 0.005

        # Scenario 1: cost in units of one infection.
        one = atkins_problem(fix["table_a2"]["scenario1"])
        r1 = atkins_solve(one)
        published = choices["scenario1"]
        # The paper's non-adaptive manager compares only the full campaign with none.
        @test expected(one, one.cost.not_not) < expected(one, one.cost.vaccinate_vaccinate)
        @test symbols(published["non_adaptive"]) == [:not, :not] == r1.plan
        # Passive: on the prior, stopping is worse than continuing and a late start
        # worse than none, so the first action is not to vaccinate, and nothing is seen.
        @test r1.act_now_action == Symbol(published["passive_first"]) == :not
        @test r1.passive == r1.no_learning ≈ r1.baseline
        @test r1.first_action == Symbol(published["active_first"]) == :vaccinate
        @test atkins_successes_needed(r1) == target(fix, "s1_successes_needed")["value"]
        @test agrees(crossover(one), fix, "s1_crossover_efficacy")
        @test r1.value_of_probing > 100                      # "over 100 units"
        @test agrees(100 * r1.value_of_probing / -r1.passive, fix, "s1_gain_percent")
        @test isapprox(r1.value_of_probing, fix["ours"]["scenario1"]["gain_units"];
                       atol=0.5)
        @test r1.no_learning <= r1.passive <= r1.active <= r1.perfect

        # Scenario 2: days of outbreak.
        two = atkins_problem(fix["table_a2"]["scenario2"])
        r2 = atkins_solve(two)
        published = choices["scenario2"]
        @test expected(two, two.cost.vaccinate_vaccinate) < expected(two, two.cost.not_not)
        @test symbols(published["non_adaptive"]) == [:vaccinate, :vaccinate]
        # Passive: wait, then vaccinate from day 7. It is the best plan fixed in advance
        # among the four campaigns, which the paper's non-adaptive manager, with two
        # campaigns to choose from, does not find.
        @test r2.plan ==
              [Symbol(published["passive_first"]),
               Symbol(published["passive_then"])] == [:not, :vaccinate]
        @test r2.act_now_action == :not && r2.passive == r2.no_learning
        @test -r2.no_learning < expected(two, two.cost.vaccinate_vaccinate)
        @test r2.first_action == Symbol(published["active_first"]) == :vaccinate
        @test atkins_successes_needed(r2) == target(fix, "s2_successes_needed")["value"]
        @test agrees(crossover(two), fix, "s2_crossover_efficacy")
        @test r2.no_learning <= r2.passive < r2.active < r2.perfect
        # The published gain of the active approach is met against not vaccinating at
        # all, not against the passive policy that the paper describes.
        ours = fix["ours"]["scenario2"]
        against_passive = r2.value_of_probing
        against_none = r2.active - r2.baseline
        @test isapprox(against_passive, ours["gain_days_against_passive"]; atol=0.1)
        @test isapprox(against_none, ours["gain_days_against_no_vaccination"]; atol=0.1)
        @test agrees(against_none, fix, "s2_gain_days")
        @test agrees(100 * against_none / -r2.baseline, fix, "s2_gain_percent")
        @test !agrees(against_passive, fix, "s2_gain_days")
        @test !agrees(100 * against_passive / -r2.passive, fix, "s2_gain_percent")

        # Section 3.4: "only one monitored vaccination" is enough to make vaccinating at
        # once the best first action. That holds here in Scenario 1; Scenario 2 needs
        # three (pinned in the fixture). With none, active and passive are the same.
        fewest = fix["ours"]["fewest_monitored_for_vaccinating_first"]
        for scenario in ("scenario1", "scenario2")
            firsts = map(0:4) do monitored
                r = atkins_solve(atkins_problem(fix["table_a2"][scenario]; monitored))
                monitored == 0 && @test r.active ≈ r.passive
                @test (r.first_action == :vaccinate) == (r.value_of_probing > 1e-9)
                return r.first_action
            end
            @test findfirst(==(:vaccinate), firsts) - 1 == fewest[scenario]
        end
    end

    @testset "de Vries et al. 2021, closing mink farms" begin
        fix = replication("devries2021_mink")
        box = fix["box2"]
        actions, states = symbols(fix["actions"]), symbols(fix["states"])
        close, open_infected = exact_number(box["cost_of_closing"]),
                               exact_number(box["cost_of_infected_open_farm"])
        farms = box["farms"]
        # One farm, in millions of euros. An open farm that is not infected costs
        # nothing; closing costs the same in both states (the fixture's "implied").
        farm(p; closing=close, infected=open_infected) = DecisionProblem(PayoffTable(actions,
                                                                                     states,
                                                                                     [-infected 0;
                                                                                      -closing -closing]),
                                                                         FiniteBelief(states,
                                                                                      [p,
                                                                                       1 -
                                                                                       p]);
                                                                         baseline=:keep_open)
        prior = parse_rational(box["probability_infected"])
        problem = farm(prior)
        values = aggregate(Expectation(), problem)
        @test values[1] == target(fix, "expected_value_keep_open_per_farm")["value"]
        @test values[2] == target(fix, "expected_value_close_per_farm")["value"]
        @test first_choice(decide(problem, MaxExpectedUtility())) == :keep_open
        @test -farms * act_now_value(problem) ==
              target(fix, "expected_loss_without_information")["value"]
        @test -farms * (act_now_value(problem) + evpi(problem)) ==
              target(fix, "expected_loss_with_perfect_information")["value"]
        @test farms * evpi(problem) == target(fix, "evpi")["value"]
        # Closed form: what is saved on the farms that would be decided wrongly.
        closed_form(p) = min(p * (open_infected - close), (1 - p) * close)
        for p in (0:20) .// 20
            @test evpi(farm(p)) == closed_form(p)
        end
        # Largest where the two actions tie, at p = closing cost / infected cost.
        @test argmax(p -> evpi(farm(p)), (0:100) .// 100) == close // open_infected ==
              2 // 5
        @test evpi(farm(2 // 5)) == 12 // 5 == 4 * evpi(problem)
        # A test of a farm with sensitivity se and specificity sp, as an experiment.
        function testing(se, sp)
            return Experiment([se 1-sp; 1-se sp]; signals=[:positive, :negative], states)
        end
        @test evsi(problem, testing(1, 1)) == evpi(problem) == 3 // 5
        @test evsi(problem, testing(1 // 2, 1 // 2)) == 0
        # Positive farms are closed: the gain is p se (10 − 4) − (1 − p)(1 − sp) 4.
        for se in (6:10) .// 10, sp in (6:10) .// 10
            gain = prior * se * (open_infected - close) - (1 - prior) * (1 - sp) * close
            @test evsi(problem, testing(se, sp)) == max(gain, 0)
            @test information_can_change_decision(problem, testing(se, sp)) == (gain > 0)
        end
        # A test that misses nothing is still worthless below a specificity of 5/6.
        @test evsi(problem, testing(1, 5 // 6)) == 0
        @test evsi(problem, testing(1, 9 // 10)) == 6 // 25
    end

    @testset "Shea et al. 2023, seventeen models of reopening" begin
        fix = replication("shea2023_covid_models")
        rows = fix["dataset_s2"]["rows"]
        @test length(rows) == 17 * 4 * 5
        series = Dict((r[1], r[2], r[3]) => (Int(r[4]), parse.(Float64, r[5:end]))
                      for r in rows)
        models = sort(unique(r[1] for r in rows))
        interventions, objectives = fix["actions"], fix["objectives"]
        @test length(models) == 17
        @test all(issorted(last(v))
                  for (k, v) in series if k[3] in ("cumu_infections",
                                                   "cumu_deaths"))
        # Sixteen groups of equal weight; G.1 and G.2 are one group.
        weight = Dict(m => (startswith(m, "G") ? 1 // 2 : 1 // 1) // 16 for m in models)
        @test sum(values(weight)) == 1
        pooled(objective, intervention, x) = sum(weight[m] *
                                                 shea_cdf(series[(m, intervention,
                                                                  objective)]..., x)
                                                 for m in models)
        function pooled_quantile(objective, intervention, q)
            lo = minimum(first(last(series[(m, intervention, objective)])) for m in models)
            hi = maximum(last(last(series[(m, intervention, objective)])) for m in models)
            for _ in 1:100
                mid = (lo + hi) / 2
                pooled(objective, intervention, mid) < q ? (lo = mid) : (hi = mid)
            end
            return hi
        end
        median(objective, intervention) = pooled_quantile(objective, intervention, 0.5)

        # The aggregate medians printed in the article.
        infections = Dict(i => median("cumu_infections", i) for i in interventions)
        @test agrees(infections["open"], fix, "infections_open_median")
        @test agrees(infections["closed"], fix, "infections_closed_median")
        @test round(Int, infections["closed"]) == 8530          # the article has 8,527
        for i in ("closed", "5percent", "2weeks")
            @test agrees(100 * (1 - infections[i] / infections["open"]), fix,
                         "infections_reduction_" * i * "_percent")
        end
        @test agrees(median("cumu_deaths", "closed"), fix, "deaths_closed_median")
        @test agrees(pooled_quantile("cumu_deaths", "closed", 0.25), fix,
                     "deaths_closed_lower_quartile")
        @test agrees(pooled_quantile("cumu_deaths", "closed", 0.75), fix,
                     "deaths_closed_upper_quartile")
        @test agrees(median("days_closed", "closed"), fix, "days_closed_closed")
        @test agrees(median("days_closed", "5percent"), fix, "days_closed_5percent_median")
        @test agrees(median("days_closed", "2weeks"), fix, "days_closed_2weeks_median")
        # The chance of needing more than the county's 200 hospital beds.
        for i in ("2weeks", "closed")
            exceed = 100 * (1 - pooled("peak_hosp", i, 200.0))
            @test agrees(exceed, fix, "exceed_200_beds_" * i * "_percent")
            @test isapprox(exceed, fix["ours"]["exceed_200_beds_" * i * "_percent"];
                           atol=0.05)
        end
        # Our pooled medians against the aggregate sheet of the dataset, within 0.5%.
        for (key, theirs) in fix["aggregate_sheet_medians"]["values"]
            intervention, objective = split(key, '|')
            objective in ("cumu_infections", "cumu_deaths", "peak_hosp") || continue
            @test isapprox(median(objective, intervention), theirs; rtol=0.005)
        end
        # The same order of the interventions for every health outcome in the pool.
        for objective in ("cumu_infections", "cumu_deaths", "peak_hosp")
            @test issorted([median(objective, i) for i in interventions])
        end
        @test issorted([median("days_closed", i) for i in interventions]; rev=true)

        # The published median probability of an outbreak under "closed" (71%) is not
        # reproduced: the submissions are of different kinds (one probability, a list
        # of zeros and ones, or a distribution), so a pooled median is not well
        # defined. Each model's probability is taken as the mean of what it submitted.
        chance = Dict(m => shea_mean(last(series[(m, "closed", "prob_outbreak")]))
                      for m in models)
        ours = fix["ours"]["probability_of_outbreak_closed"]
        @test count(m -> Set(last(series[(m, "closed", "prob_outbreak")])) ==
                         Set([0.0, 1.0]),
                    models) == 4
        @test isapprox(sum(weight[m] * chance[m] for m in models),
                       ours["weighted_mean_over_models"]; atol=0.005)
        half = ours["half_of_weight_at_or_below"]
        @test sum(weight[m] for m in models if chance[m] <= half) == 1 // 2
        @test isapprox(minimum(chance[m] for m in models if chance[m] > half),
                       ours["next_model_at"]; atol=0.01)
        published = target(fix, "probability_of_outbreak_closed_median")["value"] / 100
        @test published > ours["next_model_at"] > ours["weighted_mean_over_models"] - 0.06
        @test !agrees(100 * sum(weight[m] * chance[m] for m in models), fix,
                      "probability_of_outbreak_closed_median")

        # One payoff table for each outcome: the mean projection of each model under
        # each intervention; the states are the models.
        actions, states = symbols(interventions), Symbol.("model_", models)
        prior = FiniteBelief(states, [weight[m] for m in models])
        means(objective) = [shea_mean(last(series[(m, i, objective)]))
                            for i in interventions, m in models]
        problem(costs) = DecisionProblem(PayoffTable(actions, states, -costs), prior;
                                         baseline=Symbol(fix["baseline"]))
        worst(costs) = [interventions[argmax(view(costs, :, j))] for j in eachindex(models)]
        best(costs) = [interventions[argmin(view(costs, :, j))] for j in eachindex(models)]
        @test count(==("open"), worst(means("cumu_infections"))) ==
              target(fix, "models_ranking_open_worst_for_infections")["value"]
        # Staying closed is best in expectation for each health outcome, but not in
        # every model, so knowing which model is right has a value.
        deaths = problem(means("cumu_deaths"))
        @test first_choice(decide(deaths, MaxExpectedUtility())) == :closed
        @test count(==("closed"), best(means("cumu_deaths"))) == 14
        @test count(==("5percent"), best(means("cumu_deaths"))) == 3
        @test count(==("closed"), best(means("cumu_infections"))) == 15
        @test 0 < evpi(deaths) < -act_now_value(deaths) / 10
        @test isapprox(evpi(deaths), 25.8; atol=0.1)
        @test isapprox(evpi(problem(means("cumu_infections"))), 962; atol=1)
        # Every model agrees on the days closed, so nothing can be learned there.
        @test evpi(problem(means("days_closed"))) < 1e-9
        @test first_choice(decide(problem(means("days_closed")), MaxExpectedUtility())) ==
              :open
        # Trading deaths against days closed: the choice moves from closed to open as a
        # day of closure counts for more, and it is in between that the models matter.
        tradeoff(rate) = problem(means("cumu_deaths") .+ rate .* means("days_closed"))
        choices = [first_choice(decide(tradeoff(rate), MaxExpectedUtility()))
                   for rate in (0, 1, 5, 100)]
        @test choices == [:closed, Symbol("5percent"), :open, :open]
        @test evpi(tradeoff(3)) > 5 * evpi(tradeoff(0))
        @test evpi(tradeoff(100)) < 1e-6
    end

    @testset "Regan et al. 2006 and Rout et al. 2009, when to stop looking" begin
        fix = replication("regan2006_eradication")
        # The rule of thumb: the cost of stopping after n absent surveys, in survey
        # costs, and the n that minimises it (equation 3) or solves equation 4.
        nec(n, r, ratio) = (n - 1) + ratio * r^n
        rule(r, ratio) = argmin(n -> nec(n, r, ratio), 1:3000)
        formula(r, ratio) = log(-1 / (ratio * log(r))) / log(r)
        @test rule(0.1, 100) == target(fix, "rule_1to100_r01")["value"]
        @test rule(0.9, 100) == target(fix, "rule_1to100_r09")["value"]
        @test rule(0.1, 1000) == target(fix, "rule_1to1000_r01")["value"]
        @test rule(0.9, 1000) == target(fix, "rule_1to1000_r09")["value"]
        @test rule(0.99, 1000) == target(fix, "rule_1to1000_r099")["value"]
        @test issorted([rule(r, 100) for r in 0.1:0.1:0.9])
        # The formula is the integer rule to within one survey.
        @test all(abs(formula(r, c) - rule(r, c)) < 1
                  for r in 0.1:0.1:0.9, c in (100, 1000))
        # Where equation 4 reaches zero: r = exp(-Cs / Ce).
        crossing(ratio) = exp(-1 / ratio)
        @test abs(formula(crossing(1000), 1000)) < 1e-9
        @test agrees(crossing(1000), fix, "zero_crossing_1to1000")
        @test agrees(crossing(2), fix, "zero_crossing_1to2")
        @test round(crossing(2); digits=3) == 0.607

        # Helenium amarum, Table 1.
        t1 = fix["table1"]
        value(key, which) = Float64(exact_number(t1[key][which]))
        r_of(q, p) = value("persistence", p) * (1 - value("detection", q))
        @test rule(r_of("best", "best"), value("cost_ratio", "best")) ==
              target(fix, "helenium_best")["value"]
        @test rule(r_of("upper", "lower"), value("cost_ratio", "lower")) ==
              target(fix, "helenium_at_least")["value"]
        # The longest of the eight extreme combinations is 7, not the published 5.
        longest = maximum(rule(r_of(q, p), value("cost_ratio", c))
                          for q in ("lower", "upper"), p in ("lower", "upper"),
                              c in ("lower", "upper"))
        @test longest == fix["ours"]["helenium_upper_rule"] == 7
        @test longest == rule(r_of("lower", "upper"), value("cost_ratio", "upper"))
        @test rule(r_of("lower", "lower"), value("cost_ratio", "upper")) ==
              target(fix, "helenium_no_more_than")["value"]

        # Discounting by equations 5 and 6 as printed lengthens the search; the paper
        # reports that it shortens it.
        discounted(n, r, ratio, d) = (1 - (1 + d)^-(n - 1)) / d + ratio * r^n / (1 + d)^n
        for (ratio, id) in
            ((100, "discounted_1to100_r09"), (1000, "discounted_1to1000_r09"))
            ours = argmin(n -> discounted(n, 0.9, ratio, 0.05), 1:3000)
            @test ours == fix["ours"][id]
            @test ours > rule(0.9, ratio) > target(fix, id)["value"]
        end

        # Rout et al. 2009: the optimum as a real number, its cost, and robustness.
        ratio = Float64(exact_number(fix["rout"]["cost_ratio"]))
        r = 0.8 * (1 - 0.83)
        @test agrees(r, fix, "rout_r")
        optimum = formula(r, ratio)
        @test agrees(optimum, fix, "rout_optimal_surveys")
        @test agrees(nec(optimum, r, ratio), fix, "rout_relative_cost_at_optimum")
        @test agrees(1 / log(r), fix, "rout_one_over_log_r")
        # Robustness: the largest proportional error in the probability of presence at
        # which stopping after d surveys still costs no more than `limit` surveys.
        robustness(d, limit) = (limit - (d - 1)) / (ratio * r^d) - 1
        robust_optimal(limit) = limit + 1 / log(r) + 1                  # equation 5
        for limit in (3.0, 4.0, 6.0, 10.0)
            d = robust_optimal(limit)
            @test all(robustness(d, limit) >= robustness(d + step, limit)
                      for step in (-0.5, -0.01, 0.01, 0.5))
            @test d >= optimum                 # at least the cost-minimising number
            @test robustness(d, limit) > 0
        end
        # At the lowest admissible limit the two coincide and nothing is to spare.
        least = nec(optimum, r, ratio)
        @test robust_optimal(least) ≈ optimum
        @test abs(robustness(optimum, least)) < 1e-9

        # The dynamic programme, as adaptive_management with two states. The state is
        # the plant's status this year before the survey; stopping pays the escape cost
        # if it is present and ends the matter.
        function stops_after(p, q, cost_ratio; years=300)
            table = PayoffTable([:stop, :survey], [:present, :absent],
                                BigRat[-cost_ratio 0; -1 -1])
            moves = (; stop=BigRat[0 0; 1 1], survey=BigRat[p 0; 1-p 1])
            seen = (; survey=Experiment(BigRat[1-q 1; q 0]; signals=[:not_seen, :seen]))
            start = BigRat[p, 1 - p]           # seen last year
            solved = adaptive_management(DecisionProblem(table, start; baseline=:stop),
                                         seen; horizon=years, transitions=moves, depth=0,
                                         passive=false, method=:vectors)
            belief, surveys = start, 0
            for left in years:-1:1
                pieces = solved.value_function[left]
                pieces.actions[argmax(pieces.vectors * belief)] == :stop && return surveys
                surveys += 1
                unseen = moves.survey * (belief .* BigRat[1 - q, 1])
                belief = unseen ./ sum(unseen)
            end
            return surveys
        end
        ours = fix["ours"]
        exact(key, which) = exact_number(t1[key][which])
        for (which, q, p, c) in (("best", "best", "best", "best"),
                                 ("lower", "upper", "lower", "lower"),
                                 ("upper", "lower", "upper", "upper"))
            @test stops_after(exact("persistence", p), exact("detection", q),
                              exact("cost_ratio", c)) ==
                  ours["helenium_dynamic_programme"][which]
        end
        # Against the rule at a cost ratio of 1 : 100 (the paper's Fig. 2).
        gaps = ours["dynamic_programme_minus_rule_1to100"]
        for (key, q) in (("q_0.95", 95 // 100), ("q_0.05", 5 // 100))
            for (persistence, gap) in gaps[key]
                p = exact_number(persistence)
                @test stops_after(p, q, 100) - rule(Float64(p * (1 - q)), 100) == gap
            end
        end
        @test all(==(1), [gaps["q_0.95"][p] for p in ("0.9", "0.95", "0.98")])
        @test maximum(values(gaps["q_0.05"])) == 2          # the paper: "about five"
        # With a plant that almost never dies out, the best course is to stop at once;
        # with good detection that depends on the horizon, which the paper does not
        # state.
        stubborn = ours["persistence_0.99_1to100"]
        for (key, q) in (("q_0.95", 95 // 100), ("q_0.05", 5 // 100)),
            (years, label) in ((300, "300_years"), (600, "600_years"))
            @test stops_after(99 // 100, q, 100; years) == stubborn[key][label]
        end
        @test stubborn["q_0.05"]["300_years"] == 0 == stubborn["q_0.95"]["600_years"]
        # A longer horizon changes nothing.
        @test stops_after(8 // 10, 83 // 100, 354; years=600) == 4
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
            @test r.value == maximum(r.values) &&
                  r.chosen == r.first_stages[r.values .== r.value]
        end
        masked = DecisionProblem(PayoffTable([:a, :b, :c], [:s, :t], [9 9; 1 0; 0 1]),
                                 [1 // 2, 1 // 2]; feasible=[:b, :c])
        @test two_stage_value(masked, [:x, :y, :y]).first_stages == [:y]
        @test two_stage_value(masked, [:x, :y, :y]).value == 1
        @test_throws ArgumentError two_stage_value(p, [:a])
        @test_throws ArgumentError two_stage_value(p, [:a, :b]; partition=[1, 2])
        credal = DecisionProblem(PayoffTable([4 0; 1 2]),
                                 CredalSet([[1 // 2, 1 // 2], [1 // 4, 3 // 4]]))
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
