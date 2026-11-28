# Tests for Track 3 feature additions:
#   T3.1: Certainty-equivalent VOI (ce_evpi, ce_evppi, ce_evsi)
#   T3.2: Ecological study designs (CountStudy, OccupancyStudy, CaptureRecaptureStudy, MvNormalStudy)
#   T3.3: Antithetic Multilevel Monte Carlo (MultilevelMC)
#   T3.4: Stochastic dominance, KMM SmoothAmbiguity, Info-gap, PRIM scenario discovery
#   T3.5: budget_selection_model, exact select_under_budget, constrained_policy_model, prune_generators
#   T3.6: Time-varying adaptive_management, sample value_of_waiting / two_stage_value,
#         evim, evmi, two_way_policy_regions, decision_curve
#   T3.7: PSIS k̂ diagnostics, CRN propagation, Tables.jl on core types

using HiGHS: HiGHS
using JuMP: JuMP
using LinearAlgebra: det, inv
using Statistics: cov, mean, var

@testset "Track 3 features" begin
    @testset "T3.1: certainty-equivalent VOI (ce_evpi, ce_evppi, ce_evsi)" begin
        # Exact linear identity on rational payoff tables.
        X = PayoffTable([:safe, :risky], [:low, :high], [60 60; 20 120])
        p = DecisionProblem(X, [1 // 2, 1 // 2])
        exp_e = Experiment([4//5 1//5; 1//5 4//5]; signals=[:lo, :hi])
        for u in (IndividualsUtility(:frogs), LinearUtility(:frogs),
                  LinearUtility(:frogs; worst=0, best=100))
            r_pi = ce_evpi(p, u)
            r_si = ce_evsi(p, u, exp_e)
            r_ppi = ce_evppi(p, u, [:a, :b])
            @test r_pi.value == evpi(p) && r_pi.value isa Rational
            @test r_si.value == evsi(p, exp_e) && r_si.value isa Rational
            @test r_ppi.value == evpi(p)
            @test !r_pi.domain_limited
        end

        # CARA (ExponentialUtility): exact translation invariance (buying price == CE_info - CE_0)
        # and shift invariance at large population scales (no expm1 saturation).
        u_exp = ExponentialUtility(:frogs; risk_aversion=0.05)
        r_cara = ce_evpi(p, u_exp)
        @test r_cara.value ≈ r_cara.ce_gain atol = 1e-12
        @test r_cara.value > 0
        # Verify Howard's equation V_info(v*) == V_0 directly.
        vstar = r_cara.value
        v0_direct = maximum(0.5 * u_exp(X.Q[a, 1]) + 0.5 * u_exp(X.Q[a, 2]) for a in 1:2)
        vinf_direct = 0.5 * max(u_exp(60 - vstar), u_exp(20 - vstar)) +
                      0.5 * max(u_exp(60 - vstar), u_exp(120 - vstar))
        @test vinf_direct ≈ v0_direct atol = 1e-10
        # Large-scale shift by +1000 frogs must leave CARA CE-EVPI identical.
        p_big = DecisionProblem(PayoffTable([:safe, :risky], [:low, :high],
                                            [1060 1060; 1020 1120]), [0.5, 0.5])
        r_big = ce_evpi(p_big, u_exp)
        @test r_big.value ≈ r_cara.value rtol = 1e-10
        # Under high risk aversion (r = 0.05), an 80%-accurate test never switches the
        # manager away from :safe, so CE-EVSI is 0; at milder r = 0.01 it is strictly positive.
        @test ce_evsi(p, u_exp, exp_e).value == 0.0
        u_exp_mild = ExponentialUtility(:frogs; risk_aversion=0.01)
        r_cara_mild_pi = ce_evpi(p, u_exp_mild)
        r_cara_mild_si = ce_evsi(p, u_exp_mild, exp_e)
        @test 0 < r_cara_mild_si.value <= r_cara_mild_pi.value + 1e-12
        # Outcome counting against (best < worst).
        u_neg = ExponentialUtility(:infected; risk_aversion=0.05, worst=200, best=0)
        p_inf = DecisionProblem(PayoffTable([:none, :treat], [:mild, :bad],
                                            [10 150; 40 40]), [0.5, 0.5])
        r_neg = ce_evpi(p_inf, u_neg)
        @test r_neg.value > 0 && r_neg.informed_ce < r_neg.act_now_ce

        # CRRA (PowerUtility): verifies Howard buying price V_info(v*) == V_0 for both
        # ρ < 1 and ρ > 1, and domain_limited detection when ρ < 1 hits zero.
        for ρ in (0.5, 1.0, 2.0)
            u_pow = PowerUtility(:frogs; risk_aversion=ρ)
            rp = ce_evpi(p, u_pow)
            @test !rp.domain_limited && rp.value > 0
            v = rp.value
            v0_p = maximum(0.5 * u_pow(X.Q[a, 1]) + 0.5 * u_pow(X.Q[a, 2]) for a in 1:2)
            # In state :low, :safe is best (60 - v > 0); in :high, :risky is best (120 - v > 0).
            vinf_p = 0.5 * u_pow(60 - v) + 0.5 * u_pow(120 - v)
            @test vinf_p ≈ v0_p atol = 1e-10
            # Verify CE-EVSI also satisfies Howard's equation (using a mildly shifted table
            # where 80%-accurate test changes the choice for all three ρ).
            X_si = PayoffTable([:safe, :risky], [:low, :high], [60 60; 45 120])
            p_si = DecisionProblem(X_si, [1 // 2, 1 // 2])
            rs = ce_evsi(p_si, u_pow, exp_e)
            rp_si = ce_evpi(p_si, u_pow)
            @test 0 < rs.value <= rp_si.value + 1e-10
            vs = rs.value
            v0_si = maximum(0.5 * u_pow(X_si.Q[a, 1]) + 0.5 * u_pow(X_si.Q[a, 2])
                            for a in 1:2)
            vsi_p = sum(begin
                            gw = [0.5 * exp_e.likelihood[z, 1],
                                  0.5 * exp_e.likelihood[z, 2]]
                            maximum(gw[1] * u_pow(X_si.Q[a, 1] - vs) +
                                    gw[2] * u_pow(X_si.Q[a, 2] - vs) for a in 1:2)
                        end
                        for z in 1:2)
            @test vsi_p ≈ v0_si atol = 1e-10
        end
        # Domain-limited case for ρ < 1.
        p_lim = DecisionProblem(PayoffTable([:a, :b], [:s1, :s2, :s3],
                                            [1 100 0; 0 0 100]), [0.1, 0.45, 0.45])
        r_lim = ce_evpi(p_lim, PowerUtility(:x; risk_aversion=0.5))
        @test r_lim.domain_limited && r_lim.value === nothing && r_lim.ce_gain > 0

        # OutcomeTable applies utility per run before replicate averaging (Jensen).
        rows = [(; draw=:low, action=:safe, replicate=1, frogs=50.0),
                (; draw=:low, action=:safe, replicate=2, frogs=70.0),
                (; draw=:high, action=:safe, replicate=1, frogs=50.0),
                (; draw=:high, action=:safe, replicate=2, frogs=70.0),
                (; draw=:low, action=:risky, replicate=1, frogs=20.0),
                (; draw=:low, action=:risky, replicate=2, frogs=20.0),
                (; draw=:high, action=:risky, replicate=1, frogs=120.0),
                (; draw=:high, action=:risky, replicate=2, frogs=120.0)]
        ot = OutcomeTable(rows; outcomes=[:frogs])
        r_ot = ce_evpi(ot, PowerUtility(:frogs; risk_aversion=0.5))
        # Replicate noise on :safe reduces its CE below 60, changing CE-EVPI vs cell means.
        @test !r_ot.domain_limited && r_ot.value > 0
        @test r_ot.value != ce_evpi(p, PowerUtility(:frogs; risk_aversion=0.5)).value
        @test_throws ArgumentError ce_evpi(p, InfectionUtility())
    end

    @testset "T3.2: ecological study designs" begin
        rng = Xoshiro(302)

        # CountStudy: Poisson (conjugate Gamma) and Negative Binomial (overdispersed).
        cs_pois = CountStudy(:λ; n=10)
        for μ in (0.5, 5.0, 50.0, 500.0)
            xs = [simulate_study(rng, cs_pois, (; λ=μ)).total for _ in 1:2000]
            @test abs(mean(xs) - 10 * μ) < 4 * sqrt(10 * μ / 2000)
            @test abs(var(xs) - 10 * μ) / (10 * μ) < 0.15
        end
        # Conjugate Gamma posterior mean matches (a + total) / (b + n).
        post_c = study_posterior(rng, cs_pois, (; total=40), (; shape=2.0, rate=1.0);
                                 draws=5000)
        @test abs(mean(post_c.λ) - 42.0 / 11.0) < 0.05
        # Negative Binomial variance = n*λ + (n*λ)^2 / (n*k) = n*(λ + λ^2/k).
        cs_nb = CountStudy(:λ; n=10, dispersion=2.0)
        xs_nb = [simulate_study(rng, cs_nb, (; λ=4.0)).total for _ in 1:4000]
        @test abs(mean(xs_nb) - 40.0) < 4 * sqrt(120.0 / 4000)
        @test abs(var(xs_nb) - 120.0) / 120.0 < 0.12
        @test isfinite(study_loglikelihood(cs_nb, (; total=35), (; λ=4.0)))
        @test_throws ArgumentError study_posterior(rng, cs_nb, (; total=35),
                                                   (; shape=1, rate=1))

        # OccupancyStudy (MacKenzie et al. 2002): exact sufficient stats + EVSI.
        occ = OccupancyStudy(:ψ, :p; sites=30, visits=4)
        ψ0, p0 = 0.6, 0.4
        p_any = 1 - (1 - p0)^4
        sim_occ = [simulate_study(rng, occ, (; ψ=ψ0, p=p0)) for _ in 1:3000]
        @test abs(mean(d.sites_detected for d in sim_occ) - 30 * ψ0 * p_any) < 0.3
        @test abs(mean(d.total_detections for d in sim_occ) - 30 * ψ0 * 4 * p0) < 0.5
        # Exact log-likelihood check: probabilities over all (s_det, d_tot) at S=2, K=2 sum to 1.
        occ_small = OccupancyStudy(:ψ, :p; sites=2, visits=2)
        # Enumerate exact site outcomes (y1, y2) ∈ {0, 1, 2}^2.
        ll_probs = Dict{Tuple{Int,Int},Float64}()
        for y1 in 0:2, y2 in 0:2
            p1 = y1 == 0 ? (1 - ψ0 + ψ0 * (1 - p0)^2) :
                 ψ0 * binomial(2, y1) * p0^y1 * (1 - p0)^(2 - y1)
            p2 = y2 == 0 ? (1 - ψ0 + ψ0 * (1 - p0)^2) :
                 ψ0 * binomial(2, y2) * p0^y2 * (1 - p0)^(2 - y2)
            key = ((y1 > 0) + (y2 > 0), y1 + y2)
            ll_probs[key] = get(ll_probs, key, 0.0) + p1 * p2
        end
        # Ratio of study_loglikelihood at two parameter values matches exact log-ratio.
        ψ1, p1 = 0.3, 0.7
        for ((sd, dt), _) in ll_probs
            d = (; sites_detected=sd, total_detections=dt)
            ll0 = study_loglikelihood(occ_small, d, (; ψ=ψ0, p=p0))
            ll1 = study_loglikelihood(occ_small, d, (; ψ=ψ1, p=p1))
            # Exact probability ratio (combinatorial factor cancels!).
            exact0 = ((1 - ψ0 + ψ0 * (1 - p0)^2)^(2 - sd)) * (ψ0^sd) * (p0^dt) *
                     ((1 - p0)^(2 * sd - dt))
            exact1 = ((1 - ψ1 + ψ1 * (1 - p1)^2)^(2 - sd)) * (ψ1^sd) * (p1^dt) *
                     ((1 - p1)^(2 * sd - dt))
            @test ll1 - ll0 ≈ log(exact1 / exact0) atol = 1e-12
        end
        # End-to-end EVSI with ImportanceSampling and SummaryStatistics on OccupancyStudy.
        n_occ = 800
        ψs = 0.2 .+ 0.6 .* rand(rng, n_occ)
        ps = 0.2 .+ 0.6 .* rand(rng, n_occ)
        Q_occ = vcat(zeros(1, n_occ), permutedims(ψs .- 0.5))
        prob_occ = DecisionProblem(PayoffTable(Q_occ; actions=[:wait, :treat]),
                                   SampleBelief((; ψ=ψs, p=ps)))
        e_occ_is = evsi(prob_occ, ImportanceSampling(prob_occ, occ; datasets=120, seed=1))
        e_occ_ss = evsi(prob_occ, SummaryStatistics(prob_occ, occ; seed=1))
        @test 0 < e_occ_is.value <= evpi(prob_occ) + 2 * e_occ_is.standard_error
        @test 0 < e_occ_ss.value <= evpi(prob_occ) + 2 * e_occ_ss.standard_error

        # CaptureRecaptureStudy (closed M0 / Lincoln-Petersen).
        cr = CaptureRecaptureStudy(:N, :p; occasions=2)
        sim_cr = [simulate_study(rng, cr, (; N=100, p=0.3)) for _ in 1:3000]
        @test abs(mean(d.unique_captured for d in sim_cr) - 100 * (1 - 0.7^2)) < 0.5
        @test abs(mean(d.total_captures for d in sim_cr) - 200 * 0.3) < 0.6
        # Fixed known detection probability also works.
        cr_fix = CaptureRecaptureStudy(:N, 0.4; occasions=3)
        @test DecisionAnalysis._study_parameters(cr_fix) == [:N]
        @test study_loglikelihood(cr_fix, (; unique_captured=15, total_captures=20),
                                  (; N=10)) == -Inf
        @test isfinite(study_loglikelihood(cr_fix,
                                           (; unique_captured=15, total_captures=20),
                                           (; N=40)))

        # MvNormalStudy: exact simulation covariance, log-likelihood, and conjugate posterior.
        Σ = [1.0 0.4; 0.4 2.0]
        mvn = MvNormalStudy((:a, :b); n=10, covariance=Σ)
        sim_mvn = [simulate_study(rng, mvn, (; a=1.0, b=-2.0)) for _ in 1:4000]
        M_mat = reduce(hcat, [[d.xbar_a, d.xbar_b] for d in sim_mvn])
        @test mean(M_mat; dims=2) ≈ [1.0; -2.0;;] atol = 0.05
        @test cov(M_mat; dims=2) ≈ Σ ./ 10 rtol = 0.1
        Σ0 = [2.0 0.0; 0.0 2.0]
        post_mvn = study_posterior(rng, mvn, (; xbar_a=2.0, xbar_b=1.0),
                                   (; mean=[0.0, 0.0], covariance=Σ0); draws=5000)
        Λ_post = inv(Σ0) + 10 * inv(Σ)
        μ_post = inv(Λ_post) * (10 * inv(Σ) * [2.0, 1.0])
        @test [mean(post_mvn.a), mean(post_mvn.b)] ≈ μ_post atol = 0.03
    end

    @testset "T3.3: MultilevelMC antithetic nested EVPPI and EVSI" begin
        # Exact normal model: payoffs [0, 1 + p1 - p2], p1 ~ N(0, 1), p2 ~ N(0, 4).
        # True EVPPI(p1) = E[max(0, 1 + p1)] - 1 = voi_normal_value(1.0).
        payoffs_norm(θ) = [0.0, 1.0 + θ.p1 - θ.p2]
        sample_p1(r) = (; p1=randn(r))
        sample_p2(r, _) = (; p2=2.0 * randn(r))
        truth_evppi = voi_normal_value(1.0)
        est_mlmc = evppi_nested(payoffs_norm, sample_p1, sample_p2,
                                MultilevelMC(; tol=0.015, base_inner=2, pilot=200);
                                seed=303)
        @test est_mlmc.quantity === :evppi && est_mlmc.method === :MultilevelMC
        @test abs(est_mlmc.value - truth_evppi) < 4 * est_mlmc.standard_error + 0.02
        @test est_mlmc.details.levels >= 3
        # Antithetic level differences Δ_l are nonpositive and variances decay across levels.
        @test all(<=(0.0), est_mlmc.details.level_means[2:end])
        @test est_mlmc.details.level_variances[end] < est_mlmc.details.level_variances[1]

        # MLMC EVSI of NormalStudy(:p1; n=10, sd=1) on the same model:
        # True EVSI = voi_normal_value(sqrt(10 / 11)).
        rng = Xoshiro(304)
        n_s = 1000
        draws_s = (; p1=randn(rng, n_s), p2=2.0 .* randn(rng, n_s))
        Q_s = vcat(zeros(1, n_s), permutedims(1.0 .+ draws_s.p1 .- draws_s.p2))
        prob_s = DecisionProblem(PayoffTable(Q_s; actions=[:ref, :new]),
                                 SampleBelief(draws_s))
        design_n = NormalStudy(:p1; n=10, sd=1.0)
        truth_evsi = voi_normal_value(sqrt(10 / 11))
        est_evsi = evsi(prob_s, design_n,
                        MultilevelMC(; tol=0.02, base_inner=2, pilot=200);
                        payoffs=payoffs_norm, prior=(; mean=0.0, sd=1.0), seed=305)
        @test est_evsi.quantity === :evsi && est_evsi.method === :MultilevelMC
        @test abs(est_evsi.value - truth_evsi) < 4 * est_evsi.standard_error + 0.02
    end

    @testset "T3.4: stochastic dominance, SmoothAmbiguity, info-gap, PRIM" begin
        # FSD and SSD: exact rational arithmetic.
        # Action :safe = [50, 50], :risky = [20, 80] (mean-preserving spread of :safe!),
        # :bad = [10, 40] (FSD-dominated by both).
        tbl = PayoffTable([:safe, :risky, :bad], [:s1, :s2], [50 50; 20 80; 10 40])
        prob = DecisionProblem(tbl, [1 // 2, 1 // 2])
        @test fsd_dominates(prob, :safe, :bad)
        @test fsd_dominates(prob, :risky, :bad)
        @test !fsd_dominates(prob, :safe, :risky)
        @test !fsd_dominates(prob, :risky, :safe)
        @test ssd_dominates(prob, :safe, :risky)
        @test !ssd_dominates(prob, :risky, :safe)
        @test decide(prob, StochasticDominance(; order=1)).chosen == [:safe, :risky]
        @test decide(prob, StochasticDominance(; order=2)).chosen == [:safe]
        @test decide(prob, StochasticDominance(; order=2)).details.dominated_by[:risky] ===
              :safe

        # KMM SmoothAmbiguity over a CredalSet: interpolates between neutral and LowerExpectation.
        cred = CredalSet([:s1, :s2], [[4 // 5, 1 // 5], [1 // 5, 4 // 5]])
        cp = DecisionProblem(PayoffTable([:safe, :risky], [:s1, :s2], [50 50; 0 120]), cred)
        s_neut = aggregate(SmoothAmbiguity(0.0), cp)
        s_mod = aggregate(SmoothAmbiguity(0.05), cp)
        s_high = aggregate(SmoothAmbiguity(5.0), cp)
        lo_exp = Float64.(aggregate(LowerExpectation(), cp))
        @test s_neut ≈ [50.0, 60.0]
        @test lo_exp[2] < s_mod[2] < s_neut[2]
        @test s_high ≈ lo_exp atol = 0.2
        # Custom phi form agrees with exponential form.
        s_custom = aggregate(SmoothAmbiguity(v -> -exp(-0.05 * v);
                                             inverse=y -> -log(-y) / 0.05), cp)
        @test s_custom ≈ s_mod atol = 1e-12

        # Info-gap robustness & opportuneness: exact TV-ball rationals.
        # Nominal p0 = [1/2, 1/2]: :safe gives 50 everywhere; :risky gives [0, 120] (mean 60).
        p_ig = DecisionProblem(PayoffTable([:safe, :risky], [:s1, :s2], [50 50; 0 120]),
                               [1 // 2, 1 // 2])
        # At requirement 45: :safe has α̂ = 1 (immune); :risky drops 120 per unit mass moved,
        # so 60 - 120 α >= 45 => α <= 15/120 = 1/8.
        rob = info_gap_robustness(p_ig, 45)
        @test rob.robustness == [1 // 1, 1 // 8]
        @test rob.chosen == [:safe]
        # Opportuneness for windfall 90: :safe unattainable (nothing); :risky needs 60 + 120 β >= 90 => β = 1/4.
        opp = info_gap_opportuneness(p_ig, 90)
        @test opp.opportuneness[1] === nothing && opp.opportuneness[2] == 1 // 4
        @test opp.chosen == [:risky]
        # Callable nested uncertainty horizon.
        rob_fn = info_gap_robustness((a, α) -> a === :safe ? 50 - 10α : 60 - 100α,
                                     [:safe, :risky], 45; max_horizon=2.0)
        @test rob_fn.robustness ≈ [0.5, 0.15] atol = 1e-6
        opp_fn = info_gap_opportuneness((a, α) -> a === :safe ? 50 + 5α : 60 + 60α,
                                        [:safe, :risky], 90; max_horizon=2.0)
        @test opp_fn.opportuneness[1] === nothing
        @test opp_fn.opportuneness[2] ≈ 0.5 atol = 1e-6

        # PRIM scenario discovery: recovers a planted failure box in 3D parameter space.
        rng = Xoshiro(306)
        n_pr = 1200
        x1, x2, x3 = rand(rng, n_pr), rand(rng, n_pr), rand(rng, n_pr)
        b_pr = SampleBelief((; x1, x2, x3))
        failed = (x1 .> 0.6) .& (x2 .< 0.4)
        res_prim = prim_boxes(b_pr, failed; peel_alpha=0.05, min_mass=0.05)
        @test res_prim.base_rate ≈ mean(failed) atol = 1e-12
        @test !isempty(res_prim.frontier)
        # Some box on the Pareto frontier has high density (>0.85) and high coverage (>0.7).
        @test any(b -> b.density > 0.85 && b.coverage > 0.70, res_prim.frontier)
    end

    @testset "T3.5: budget_selection_model, exact knapsack, constrained_policy_model, prune_generators" begin
        # Classic knapsack counterexample where greedy value/cost fails:
        # Budget = 100; units: (v=61, c=60) [ratio 1.017], (v=50, c=50) [1.0], (v=50, c=50) [1.0].
        # Greedy picks unit 1 (value 61); exact picks units 2 & 3 (value 100)!
        units = [(; choice=:act_now, value=61, cost=60, largest_cost=60),
                 (; choice=:act_now, value=50, cost=50, largest_cost=50),
                 (; choice=:act_now, value=50, cost=50, largest_cost=50)]
        g = select_under_budget(units, 100; method=:greedy)
        ex = select_under_budget(units, 100; method=:exact)
        @test g.value == 61 && ex.value == 100 && ex.selected == [2, 3]
        # LP relaxation via budget_selection_model solved exactly by internal simplex.
        prog_lp = budget_selection_model(units, 100; integer=false)
        sol_lp = solve_program(prog_lp)
        @test sol_lp.status === :optimal && sol_lp.exact
        @test sol_lp.objective == 61 + 50 * (40 // 50) # 101//1
        # MILP via HiGHS matches exact branch-and-bound.
        prog_milp = budget_selection_model(units, 100; integer=true)
        JuMP.set_optimizer(prog_milp.model, HiGHS.Optimizer)
        JuMP.set_silent(prog_milp.model)
        JuMP.optimize!(prog_milp.model)
        @test JuMP.objective_value(prog_milp.model) ≈ 100.0 atol = 1e-8

        # constrained_policy_model: expectation and chance constraints in exact rationals.
        Q_u = PayoffTable([:cheap, :mid, :expensive], [:s1, :s2], [10 10; 40 20; 100 0])
        C_tbl = PayoffTable([:cheap, :mid, :expensive], [:s1, :s2], [0 0; 20 20; 50 50])
        pr_c = DecisionProblem(Q_u, [1 // 2, 1 // 2])
        # Max EU subject to expected cost <= 25: mix 1/2 :cheap + 1/2 :expensive gives EU=30,
        # whereas :mid alone has cost 20 <= 25 and EU = 30! Add chance constraint P(Q >= 10) >= 3/4:
        # P(Q>=10) is [1, 1, 1/2]. Pure :expensive has EU=50, cost=50, prob=1/2.
        # Mix 1/2 :mid + 1/2 :expensive has cost=35, prob=3/4, EU=40.
        prog_cp = constrained_policy_model(pr_c; constraints=[(C_tbl, :<=, 35)],
                                           chance=[(10, 3 // 4)])
        sol_cp = solve_program(prog_cp)
        @test sol_cp.status === :optimal && sol_cp.exact
        @test sol_cp.values.μ == Rational{BigInt}[0 // 1, 1 // 2, 1 // 2]
        @test sol_cp.objective == 40 // 1

        # prune_generators: removes interior and duplicate points from a 3-state CredalSet.
        v1 = [1 // 1, 0 // 1, 0 // 1]
        v2 = [0 // 1, 1 // 1, 0 // 1]
        v3 = [0 // 1, 0 // 1, 1 // 1]
        mid1 = [1 // 3, 1 // 3, 1 // 3]
        edge = [1 // 2, 1 // 2, 0 // 1]
        cred_big = CredalSet([:s1, :s2, :s3], [v1, mid1, v2, edge, v1, v3])
        pruned = prune_generators(cred_big)
        @test length(extreme_points(pruned)) == 3
        @test Set(extreme_points(pruned)) == Set([v1, v2, v3])
    end

    @testset "T3.6: time-varying adaptive_management, sample waiting/two-stage, evim, evmi, 2-way, decision_curve" begin
        # Time-varying adaptive_management: 2 periods with different payoffs and experiments,
        # cross-checked against brute-force hand calculation.
        Q1 = PayoffTable([:safe, :probe], [:bad, :good], [0 0; -1 1])
        Q2 = PayoffTable([:safe, :probe], [:bad, :good], [0 0; -4 6])
        e1 = Experiment([1//1 0; 0 1]; signals=[:b, :g])
        e2 = Experiment([1//2 1//2; 1//2 1//2]; signals=[:x, :y])
        p_tv = DecisionProblem(Q1, [1 // 2, 1 // 2]; baseline=:safe)
        r_tv_b = adaptive_management(p_tv, [(; probe=e1), (; probe=e2)];
                                     payoffs=[Q1, Q2], horizon=2, method=:beliefs)
        r_tv_v = adaptive_management(p_tv, [(; probe=e1), (; probe=e2)];
                                     payoffs=[Q1, Q2], horizon=2, method=:vectors)
        # Period 1: :probe pays 0 and reveals exact state; period 2 then gets 0 if :bad, 6 if :good -> active = 3!
        # No learning: period 1 best is :safe (0), period 2 best is :probe (1) -> no_learning = 1!
        @test r_tv_b.no_learning == 1 // 1
        @test r_tv_b.active == 3 // 1
        @test r_tv_b.perfect == 7 // 2 # 1/2*(0+0) + 1/2*(1+6) = 7/2
        @test r_tv_b.plan == [:safe, :probe]
        @test r_tv_v.active == r_tv_b.active && r_tv_v.no_learning == r_tv_b.no_learning
        @test r_tv_b.no_learning <= r_tv_b.passive <= r_tv_b.active <= r_tv_b.perfect

        # Sample-based value_of_waiting and two_stage_value.
        rng = Xoshiro(307)
        n_w = 1000
        x = randn(rng, n_w)
        Q_now = PayoffTable(vcat(zeros(1, n_w), permutedims(x)); actions=[:wait, :act])
        Q_late = PayoffTable(vcat(zeros(1, n_w), permutedims(x .- 0.1));
                             actions=[:wait, :act])
        p_w = DecisionProblem(Q_now, SampleBelief((; x)))
        vw = value_of_waiting(p_w, Q_late, GaussianApproximation(:x; n0=1, n=50); seed=1)
        @test vw.delay_cost >= 0 && vw.evsi_delayed.value > 0
        @test vw.value ≈ vw.evsi_delayed.value - vw.delay_cost atol = 1e-12
        @test vw.standard_error > 0

        # Sample-based two_stage_value matches finite two_stage_value when focal identifies state.
        Q_2s = PayoffTable([:aa, :ab, :bb, :ba], [:s1, :s2], [-4 -9; -5 -6; -8 -3; -8 -7])
        p_2s_fin = DecisionProblem(Q_2s, [1 // 2, 1 // 2])
        p_2s_smp = DecisionProblem(Q_2s, SampleBelief((; st=[:s1, :s2])))
        ts_fin = two_stage_value(p_2s_fin, [:a, :a, :b, :b])
        ts_smp = two_stage_value(p_2s_smp, [:a, :a, :b, :b], :st, Grouping())
        @test ts_smp.value ≈ Float64(ts_fin.value) atol = 1e-12
        @test ts_smp.gain ≈ Float64(ts_fin.gain) atol = 1e-12

        # evim: value of perfect and specific implementation (exact rationals).
        p_im = DecisionProblem(PayoffTable([:old, :new], [:s1, :s2], [10 10; 0 40]),
                               [1 // 2, 1 // 2])
        # EU = [10, 20]; 60% use :old, 40% use :new => current = 14, optimal = 20, EVPIM = 6.
        im_perf = evim(p_im; implementation=(; old=3 // 5, new=2 // 5))
        @test im_perf.value == 6 // 1 && im_perf.value isa Rational
        im_spec = evim(p_im; implementation=[3 // 5, 2 // 5], target=[1 // 5, 4 // 5])
        @test im_spec.value == 4 // 1
        ex_im = Experiment([4//5 1//5; 1//5 4//5])
        im_both = evim(p_im, ex_im; implementation=[3 // 5, 2 // 5])
        @test im_both.value == im_perf.value + evsi(p_im, ex_im)

        # evmi: structural model ensemble and approximate-vs-truth.
        m1 = DecisionProblem(PayoffTable([:a, :b], [:s], reshape([10, 2], 2, 1)), [1 // 1])
        m2 = DecisionProblem(PayoffTable([:a, :b], [:s], reshape([0, 12], 2, 1)), [1 // 1])
        em_ens = evmi([m1, m2], [1 // 2, 1 // 2])
        # Avg EU = [5, 7] (max 7); knowing model gives 0.5*10 + 0.5*12 = 11 => EVMI = 4.
        @test em_ens.value == 4 // 1 && em_ens.chosen_without == [:b]
        em_pair = evmi(m1, m2) # m1 picks :a, which gets 0 under m2 vs optimal 12 => EVMI = 12.
        @test em_pair.value == 12 // 1 && em_pair.approximate_action === :a

        # two_way_policy_regions.
        base2 = DecisionProblem(PayoffTable([:a, :b, :c], [:s], reshape([0, 0, 0], 3, 1)),
                                [1.0])
        d1 = PayoffTable([:a, :b, :c], [:s], reshape([1, -1, 0], 3, 1))
        d2 = PayoffTable([:a, :b, :c], [:s], reshape([0, 0, 2], 3, 1))
        tw = two_way_policy_regions(base2, d1, d2; range1=(-1, 1), range2=(-1, 1),
                                    grid=(11, 11))
        @test size(tw.action_grid) == (11, 11) && length(tw.rows) == 121
        @test sum(values(tw.shares)) ≈ 1.0
        @test tw.action_grid[11, 1] === :a && tw.action_grid[1, 1] === :b &&
              tw.action_grid[6, 11] === :c

        # decision_curve (Vickers & Elkin 2006) — exact rationals!
        y_dc = [1, 1, 0, 0]
        p_dc = [9 // 10, 6 // 10, 4 // 10, 1 // 10]
        dc = decision_curve(y_dc, p_dc; thresholds=[1 // 2])
        row = only(dc)
        # At pt = 1/2, odds = 1: TP/N = 2/4 = 1/2, FP/N = 0 => NB = 1/2; treat_all = 1/2 - 1/2 = 0.
        @test row.net_benefit == 1 // 2 && row.treat_all == 0 // 1 &&
              row.interventions_avoided == 1 // 2
    end

    @testset "T3.7: PSIS k̂ diagnostics, CRN propagation, and Tables.jl" begin
        # PSIS k̂ diagnostic distinguishes well-behaved vs degenerate importance weights.
        rng = Xoshiro(308)
        n_is = 1000
        θ = randn(rng, n_is)
        Q_is = vcat(zeros(1, n_is), permutedims(θ))
        prob_is = DecisionProblem(PayoffTable(Q_is; actions=[:a, :b]), SampleBelief((; θ)))
        # Mild study (sd = 1, n = 2) -> low Pareto k̂.
        st_mild = ImportanceSampling(prob_is, NormalStudy(:θ; n=2, sd=1.0); datasets=40,
                                     seed=1)
        e_mild = evsi(prob_is, st_mild)
        @test length(e_mild.details.pareto_k) == 40
        @test e_mild.details.median_pareto_k < 0.7
        # Extreme concentrated study (sd = 0.02, n = 50) -> degenerate weights, high k̂.
        st_sharp = ImportanceSampling(prob_is, NormalStudy(:θ; n=50, sd=0.02); datasets=20,
                                      seed=1)
        e_sharp = evsi(prob_is, st_sharp)
        @test e_sharp.details.max_pareto_k > 0.7
        @test e_sharp.details.high_pareto_k_fraction > 0.5

        # First-class CRN noise_correlation propagation from OutcomeTable -> PayoffTable -> evpi/Report.
        shared = [(; draw=d, action=a, replicate=r,
                   n=10.0 * d + 4.0 * sin(7.0 * d + r) + 0.01 * (a == :b))
                  for d in 1:10 for a in (:a, :b) for r in 1:8]
        ot_crn = OutcomeTable(shared; outcomes=[:n])
        pt_crn = payoff_table(ot_crn, IndividualsUtility(:n))
        @test pt_crn.noise_correlation !== nothing && pt_crn.noise_correlation > 0.99
        prob_crn = DecisionProblem(ot_crn, IndividualsUtility(:n))
        est_auto = evpi(prob_crn, SampleAverage(); seed=1)
        est_indep = evpi(prob_crn, SampleAverage(); seed=1, noise_correlation=0.0)
        @test est_auto.details.noise_correlation == pt_crn.noise_correlation
        @test est_auto.bias.cells < 0.2 * est_indep.bias.cells
        rep_crn = DecisionReport(prob_crn; seed=1)
        @test rep_crn.monte_carlo.evpi.details.noise_correlation == pt_crn.noise_correlation

        # Tables.jl support on PayoffTable, RegretTable, OutcomeTable, VOIEstimate.
        @test Tables.istable(pt_crn) && length(collect(Tables.rows(pt_crn))) == 20
        @test haskey(first(Tables.rows(pt_crn)), :standard_error)
        rt = regret_table(prob_crn)
        @test Tables.istable(rt) && length(collect(Tables.rows(rt))) == 20
        @test Tables.istable(ot_crn)
        ot_roundtrip = OutcomeTable(ot_crn; outcomes=outcome_names(ot_crn), status=:status)
        @test ot_roundtrip.values == ot_crn.values && ot_roundtrip.status == ot_crn.status
        @test Tables.istable(est_auto) && length(collect(Tables.rows(est_auto))) == 1
        @test only(Tables.rows(est_auto)).value == est_auto.value
    end
end
