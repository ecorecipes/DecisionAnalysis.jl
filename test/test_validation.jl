# Cross-language validation. The fixtures were written by the scripts of `validation/`
# (R with the package `voi`; Python with scipy), which share no code with this package;
# the tests only read them, so the suite needs neither R nor Python. Each fixture
# records its provenance (versions, script, seed). See validation/README.md.

const VALIDATION = joinpath(@__DIR__, "fixtures", "validation")
const VOI_FIXTURES = joinpath(@__DIR__, "fixtures", "voi")

fixture_rationals(v) = BigRat[parse_rational(x) for x in v]
fixture_matrix(rows) = permutedims(reduce(hcat, [fixture_rationals(r) for r in rows]))

function read_numeric_csv(path)
    lines = readlines(path)
    names = Symbol.(strip.(split(lines[1], ","), '"'))
    data = [parse.(Float64, split(line, ",")) for line in lines[2:end]]
    return NamedTuple{Tuple(names)}(Tuple([row[j] for row in data]
                                          for j in eachindex(names)))
end

@testset "cross-language validation" begin
    @testset "finite rules against Python (exact) and scipy linprog" begin
        reference = JSON.parsefile(joinpath(VALIDATION, "finite_reference.json"))
        @test occursin("linprog", reference["provenance"]["solver"])
        @test length(reference["problems"]) >= 30
        for ex in reference["problems"]
            Q = fixture_matrix(ex["payoff"])
            prior = fixture_rationals(ex["prior"])
            problem = DecisionProblem(PayoffTable(Q), prior)
            exact(key) = fixture_rationals(ex[key])

            # Textbook criteria, exactly.
            @test aggregate(WorstCase(), problem) == exact("maximin")
            @test aggregate(BestCase(), problem) == exact("maximax")
            @test aggregate(Laplace(), problem) == exact("laplace")
            @test max_regret(problem) == exact("max_regret")
            @test decide(problem, Maximin()).chosen ==
                  action_labels(problem.payoff)[argmax_set(exact("maximin"))]
            @test decide(problem, MinimaxRegret()).chosen ==
                  action_labels(problem.payoff)[argmin_set(exact("max_regret"))]
            # Hurwicz: α is the weight on the best outcome, here and in the fixture.
            for (α, scores) in zip(fixture_rationals(ex["hurwicz_optimism"]), ex["hurwicz"])
                @test aggregate(Hurwicz(α), problem) == fixture_rationals(scores)
            end
            @test aggregate(Expectation(), problem) == exact("expected_utility")
            @test expected_regret(problem) == exact("expected_regret")
            @test evpi(problem) == parse_rational(ex["evpi"])
            for (level, quantiles, cvars) in zip(fixture_rationals(ex["levels"]),
                                                 ex["quantile"], ex["cvar_linprog"])
                @test aggregate(Quantile(level), problem) == fixture_rationals(quantiles)
                # CVaR against the Rockafellar-Uryasev programme solved by scipy.
                @test Float64.(aggregate(CVaR(level), problem)) ≈ Float64.(cvars) atol = 1e-7
            end

            # The credal set: every programme against scipy's linprog.
            points = [fixture_rationals(p) for p in ex["credal"]]
            credal = DecisionProblem(PayoffTable(Q), CredalSet(points))
            actions = action_labels(credal.payoff)
            @test Float64.(aggregate(LowerExpectation(), credal)) ≈
                  Float64.(ex["lower_expectation_linprog"]) atol = 1e-7
            @test Float64.(aggregate(UpperExpectation(), credal)) ≈
                  Float64.(ex["upper_expectation_linprog"]) atol = 1e-7
            @test decide(credal, EAdmissible()).chosen ==
                  actions[Bool.(ex["e_admissible_linprog"])]
            @test decide(credal, Maximal()).chosen == actions[Bool.(ex["maximal"])]
            @test decide(credal, IntervalDominance()).chosen ==
                  actions[Bool.(ex["interval_dominance"])]
            robust = robust_evpi(credal)
            @test robust.lower == parse_rational(ex["evpi_lower"])
            @test Float64(robust.upper) ≈ ex["evpi_upper_linprog"] atol = 1e-7

            # Experiments: EVSI exactly; garbling and deficiency against linprog.
            experiments = Dict(e["name"] => Experiment(fixture_matrix(e["likelihood"]))
                               for e in ex["experiments"])
            for e in ex["experiments"]
                @test evsi(problem, experiments[e["name"]]) == parse_rational(e["evsi"])
            end
            for pair in ex["pairs"]
                garbled, of = experiments[pair["garbled"]], experiments[pair["of"]]
                @test (is_garbling_of(garbled, of) !== nothing) ==
                      pair["is_garbling_linprog"]
                @test Float64(deficiency(of, garbled)) ≈ pair["deficiency_linprog"] atol = 1e-7
            end
        end
    end

    @testset "CVaR and quantile against the R package cvar" begin
        reference = JSON.parsefile(joinpath(VALIDATION, "cvar_reference.json"))
        @test haskey(reference["provenance"], "cvar")
        examples = Dict(ex["name"] => ex for ex in load_examples())
        @test length(reference["rows"]) ==
              4 * sum(length(ex["payoff"])
                      for ex in values(examples))
        for row in reference["rows"]
            ex = examples[row["example"]]
            problem = DecisionProblem(PayoffTable(rational_matrix(ex["payoff"])),
                                      rational_vector(ex["prior"]))
            a, level = row["action"], rationalize(BigInt, row["level"])
            # cvar works with losses (positive numbers for bad tails); this package
            # with utilities: the two differ by the sign. Its expected shortfall is a
            # numerical integral, hence the tolerance.
            @test Float64(aggregate(CVaR(level), problem)[a]) ≈ -row["expected_shortfall"] rtol = 1e-4 atol = 1e-4
            @test aggregate(Quantile(level), problem)[a] == -row["value_at_risk"]
        end
    end

    @testset "sample-based estimators against the R package voi" begin
        reference = JSON.parsefile(joinpath(VOI_FIXTURES, "voi_reference.json"))
        @test reference["provenance"]["voi"] == "1.0.3"
        @test startswith(reference["provenance"]["R"], "R version 4.6.0")
        # The same 2000 draws on both sides (validation/make_sample.jl).
        s = read_numeric_csv(joinpath(VOI_FIXTURES, "normal_sample.csv"))
        n = length(s.p1)
        @test n == 2000
        function sample_problem(new)
            return DecisionProblem(PayoffTable(permutedims(hcat(zeros(n), new));
                                               actions=[:reference, :new]),
                                   SampleBelief(s))
        end

        # The normal model of the voi vignette: payoffs 0 and p1 − p2.
        normal = sample_problem(s.p1 .- s.p2)
        r = reference["normal"]
        @test evpi(normal, SampleAverage()).value ≈ r["evpi"] rtol = 1e-10
        # Strong and Oakley, and Sadatsafavi with one and two separators: the same
        # algorithm on the same draws, equal to rounding.
        @test evppi(normal, :p1, StrongOakley(; blocks=20)).value ≈ r["p1_so_20"] rtol = 1e-9
        @test evppi(normal, :p2, StrongOakley(; blocks=20)).value ≈ r["p2_so_20"] rtol = 1e-9
        @test evppi(normal, :p1, StrongOakley(; blocks=50)).value ≈ r["p1_so_50"] rtol = 1e-9
        @test evppi(normal, :p1, Sadatsafavi()).value ≈ r["p1_sal_1"] rtol = 1e-9
        @test evppi(normal, :p2, Sadatsafavi()).value ≈ r["p2_sal_1"] rtol = 1e-9
        @test evppi(normal, :p1, Sadatsafavi(; separators=2)).value ≈ r["p1_sal_2"] rtol = 1e-9
        @test evppi(normal, :p2, Sadatsafavi(; separators=2)).value ≈ r["p2_sal_2"] rtol = 1e-9
        # Regressions: different software fits the smooths (GAM.jl and mgcv; our
        # Gaussian process and voi's), so agreement is close, not exact. Observed
        # relative differences: GAM 0.02% and 0.03%, Gaussian process 0.9% and 0.5%.
        @test evppi(normal, :p1, GAMRegression()).value ≈ r["p1_gam"] rtol = 0.01
        @test evppi(normal, :p2, GAMRegression()).value ≈ r["p2_gam"] rtol = 0.01
        gp = GaussianProcessRegression(; max_points=2000)
        @test evppi(normal, :p1, gp).value ≈ r["p1_gp"] rtol = 0.03
        @test evppi(normal, :p2, gp).value ≈ r["p2_gp"] rtol = 0.03
        # Both parameters are everything: every method returns the plain EVPI.
        @test evppi(normal, (:p1, :p2), GAMRegression(; tensor=true)).value ≈
              r["p1_p2_gam"] rtol = 1e-6
        @test evppi(normal, (:p1, :p2), gp).value ≈ r["p1_p2_gp"] rtol = 1e-6
        # The built-in spline stands in for the GAM by default: within its standard
        # error of voi's GAM (observed difference 0.0064, standard error 0.0155).
        spline = evppi(normal, :p1)
        @test abs(spline.value - r["p1_gam"]) < spline.standard_error

        # EVSI of m observations of p1: each side simulates its own data, so the
        # estimates agree within Monte Carlo error; all are estimates of the closed
        # forms 0.0723251 and 0.0821176 on a sample whose EVPPI of p1 is itself 0.069 to
        # 0.084 by the methods above.
        for m in (10, 100)
            truth = voi_normal_value(sqrt(m / (m + 1)))
            xbar = s.p1 .+ randn(Xoshiro(m), n) ./ sqrt(m)
            regression = evsi(normal, SummaryStatistics((; xbar)))
            study = importance_loglikelihoods((rng, θ) -> θ.p1 + randn(rng) / sqrt(m),
                                              (x, θ) -> -m * (x - θ.p1)^2 / 2,
                                              normal.belief; datasets=1000, seed=m)
            importance = evsi(normal, study)
            approximation = evsi(normal, GaussianApproximation(:p1; n0=1, n=m))
            for e in (regression, importance, approximation)
                @test abs(e.value - r["evsi_p1_n$(m)_gam"]) < 2 * e.standard_error
                @test abs(e.value - r["evsi_p1_n$(m)_is"]) < 2 * e.standard_error
                @test abs(e.value - truth) < 2 * e.standard_error
            end
        end

        # Six parameters, payoffs 0 and p1 − p2 + (x3 + x4 + x5 + x6) / 2: voi chooses
        # its Gaussian process for five focal parameters, as `evppi` does here.
        six = sample_problem(s.p1 .- s.p2 .+ 0.5 .* (s.x3 .+ s.x4 .+ s.x5 .+ s.x6))
        r = reference["six"]
        @test r["p1_x3_x4_x5_x6_default"] == r["p1_x3_x4_x5_x6_gp"]
        @test evpi(six, SampleAverage()).value ≈ r["evpi"] rtol = 1e-10
        default = evppi(six, (:p1, :x3, :x4, :x5, :x6))
        @test default.method === :GaussianProcessRegression
        @test default.value ≈ r["p1_x3_x4_x5_x6_default"] rtol = 0.03
        @test evppi(six, (:p1, :x3, :x4, :x5, :x6), gp).value ≈
              r["p1_x3_x4_x5_x6_gp"] rtol = 0.03
        @test evppi(six, (:p2, :x3, :x4, :x5, :x6), gp).value ≈
              r["p2_x3_x4_x5_x6_default"] rtol = 0.03
        @test evppi(six, (:p1, :p2, :x3, :x4, :x5, :x6), gp).value ≈ r["all_six_gp"] rtol = 1e-6

        # The chemotherapy example bundled with voi (net benefit in pounds: a published
        # validation target, used nowhere else). The reference values are recorded; the
        # data stay in the R package, so the comparison itself runs only with
        # DECISIONANALYSIS_R=1 and R with voi installed.
        r = reference["chemotherapy"]
        @test r["evpi"] ≈ 368.6051 atol = 5e-5       # the value printed in voi's vignette
        if get(ENV, "DECISIONANALYSIS_R", "") == "1"
            dir = mktempdir()
            script = joinpath(@__DIR__, "..", "validation", "chemo_data.R")
            @test strip(read(`Rscript $script $dir`, String)) ==
                  reference["provenance"]["voi"]
            nb = read_numeric_csv(joinpath(dir, "chemo_nb.csv"))
            pars = read_numeric_csv(joinpath(dir, "chemo_pars.csv"))
            Q = permutedims(hcat(nb.SoC, nb.Novel))
            @test size(Q, 2) == r["draws"]
            chemo = DecisionProblem(PayoffTable(Q; actions=[:SoC, :Novel]),
                                    SampleBelief(pars))
            @test evpi(chemo, SampleAverage()).value ≈ r["evpi"] rtol = 1e-9
            for name in (:p_side_effects_t1, :logor_side_effects)
                @test evppi(chemo, name, StrongOakley(; blocks=50)).value ≈
                      r["$(name)_so_50"] rtol = 1e-8
                @test evppi(chemo, name, Sadatsafavi()).value ≈ r["$(name)_sal_1"] rtol = 1e-8
                @test evppi(chemo, name, Sadatsafavi(; separators=2)).value ≈
                      r["$(name)_sal_2"] rtol = 1e-8
                # GAM.jl against mgcv: within a fraction of the standard error
                # (observed differences 0.60 and 2.15 pounds, standard errors 5.7, 5.6).
                g = evppi(chemo, name, GAMRegression())
                @test abs(g.value - r["$(name)_gam"]) < 0.5 * g.standard_error
            end
            @test evppi(chemo, :u_hospital, Sadatsafavi()).value ≈ r["u_hospital_sal_1"] rtol = 1e-8
            g = evppi(chemo, :u_hospital, GAMRegression())
            @test abs(g.value - r["u_hospital_gam"]) < 0.5 * g.standard_error
            pair = evppi(chemo, (:p_side_effects_t1, :logor_side_effects),
                         GAMRegression(; tensor=true))
            @test abs(pair.value - r["pair_gam"]) < 0.5 * pair.standard_error
            # Gaussian process on the first 1500 draws on both sides (see
            # validation/voi_reference.R): observed differences 1.3 and 0.1 pounds.
            first = 1:1500
            head = DecisionProblem(PayoffTable(Q[:, first]; actions=[:SoC, :Novel]),
                                   SampleBelief(map(c -> c[first], pars)))
            one = evppi(head, :logor_side_effects, GaussianProcessRegression())
            @test abs(one.value - r["logor_side_effects_gp_first1500"]) <
                  0.5 * one.standard_error
            group = (:p_side_effects_t1, :p_side_effects_t2, :logor_side_effects,
                     :p_hospitalised_total, :p_died)
            five = evppi(head, group)
            @test five.method === :GaussianProcessRegression
            @test abs(five.value - r["group_of_five_gp_first1500"]) <
                  0.5 * five.standard_error
        end
    end
end
