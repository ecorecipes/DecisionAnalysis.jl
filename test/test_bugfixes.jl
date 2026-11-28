# Regression tests for bug fixes B1-B23 (static audit at commit f029dd8).

using CDDLib: CDDLib
using GAM: GAM
using HiGHS: HiGHS
using JuMP: JuMP
using Statistics: cor

struct _CountModel
    val::Float64
    wp_calls::Ref{Int}
    aa_calls::Ref{Int}
end
function DecisionAnalysis.with_parameters(m::_CountModel, θ)
    m.wp_calls[] += 1
    return _CountModel(θ.x, m.wp_calls, m.aa_calls)
end
function DecisionAnalysis.apply_action(m::_CountModel, a::Float64)
    m.aa_calls[] += 1
    return _CountModel(m.val + a, m.wp_calls, m.aa_calls)
end
DecisionAnalysis.simulate(m::_CountModel; rng) = m.val + 0.1 * randn(rng)
DecisionAnalysis.outcome(m::_CountModel, s::Float64) = (; y=s)

@testset "bugfix regressions" begin
    @testset "B1: GaussianProcessRegression rank deficiency and small hyper_points" begin
        rng = Xoshiro(20261127)
        n = 40
        x1 = randn(rng, n)
        # Collinear columns passed directly to fitted_conditional_means must terminate.
        cols = (; a=x1, b=2.0 .* x1 .+ 1.0, c=randn(rng, n))
        Q = vcat(zeros(1, n), permutedims(x1 .+ 0.5 .* cols.c .+ 0.1 .* randn(rng, n)))
        w = fill(1 / n, n)
        G = fitted_conditional_means(GaussianProcessRegression(; hyper_points=10), cols, Q,
                                     w)
        @test size(G) == size(Q)
        @test all(isfinite, G)
        # p >= hyper_points - 3 (e.g. p = 8, hyper_points = 10) must not degenerate to -Inf.
        X8 = [randn(rng, 40) for _ in 1:8]
        cols8 = NamedTuple{Tuple(Symbol(:x, i) for i in 1:8)}(Tuple(X8))
        Q8 = vcat(zeros(1, 40), permutedims(2.0 .* X8[1] .- X8[2] .+ 0.1 .* randn(rng, 40)))
        G8 = fitted_conditional_means(GaussianProcessRegression(; hyper_points=10,
                                                                max_points=40), cols8, Q8,
                                      w)
        @test all(isfinite, G8)
        @test cor(vec(G8[2, :]), vec(Q8[2, :])) > 0.8
    end

    @testset "B2: MultiplicativeUtility interaction constant near sum(k) = 1" begin
        u1 = LinearUtility(:a; worst=0, best=10)
        u2 = LinearUtility(:b; worst=0, best=10)
        u3 = LinearUtility(:c; worst=0, best=10)
        # Exact n = 2 closed form, including tiny offsets around sum = 1.
        for δ in (-1e-12, -1e-6, 1e-12, 1e-6)
            k1, k2 = 0.5, 0.5 + δ
            m = MultiplicativeUtility(u1 => k1, u2 => k2)
            @test isfinite(m.interaction)
            @test m.interaction ≈ (1 - k1 - k2) / (k1 * k2) rtol = 1e-10
        end
        # n = 3 with tiny offsets around sum(k) = 1: must not hang, return NaN, or return -1.
        for δ in (-1e-14, -1e-8, 1e-14, 1e-8)
            ks = (0.3, 0.3, 0.4 + δ)
            m = MultiplicativeUtility(u1 => ks[1], u2 => ks[2], u3 => ks[3])
            K = m.interaction
            @test isfinite(K) && K > -1.0
            if δ < 0
                @test K > 0
            else
                @test -1.0 < K < 0
            end
            # Check factored polynomial g(K) = (Π(1 + K k_i) - 1 - K) / K ≈ 0.
            e2 = ks[1] * ks[2] + ks[1] * ks[3] + ks[2] * ks[3]
            e3 = ks[1] * ks[2] * ks[3]
            @test abs((sum(ks) - 1) + e2 * K + e3 * K^2) < 1e-12
        end
    end

    @testset "B3: MomentMatching preserves multi-parameter prior correlation and Loewner cap" begin
        rng = Xoshiro(99)
        n = 1200
        z = randn(rng, n)
        # Strongly correlated focal parameters: a ≈ b, so a - b ≈ 0 under the joint prior,
        # whereas independent marginals would make a - b have variance 2!
        a = z
        b = z .+ 0.05 .* randn(rng, n)
        belief = SampleBelief((; a, b))
        payoffs(θ) = [0.0, θ.a - θ.b, 0.5 * (θ.a - θ.b)]
        Q = reduce(hcat, [payoffs((; a=a[i], b=b[i])) for i in 1:n])
        prob = DecisionProblem(PayoffTable(Q; actions=[:base, :act1, :act2]), belief)
        evppi_val = evppi(prob, (:a, :b), SplineRegression(; knots=1, degree=1)).value
        study = MomentMatching((:a, :b); payoffs, nested=20,
                               regression=SplineRegression(; knots=1, degree=1),
                               simulate_data=(r, θ) -> θ.a + 0.1 * randn(r),
                               posterior=(r, d) -> (; a=d .+ 0.1 .* randn(r, 200),
                                                    b=d .+ 0.1 .* randn(r, 200)))
        est = evsi(prob, study; seed=1)
        @test est.value <= evppi_val + 1e-8
        # Early-return zero-variance branch exposes full details schema.
        flat_prob = DecisionProblem(PayoffTable(zeros(2, n); actions=[:base, :act1]),
                                    belief)
        flat_est = evsi(flat_prob,
                        MomentMatching((:a, :b); payoffs=θ -> [0.0, 0.0], nested=10,
                                       simulate_data=(r, θ) -> 0.0,
                                       posterior=(r, d) -> (; a=zeros(10), b=zeros(10)));
                        seed=1)
        @test flat_est.value == 0.0
        @test haskey(flat_est.details, :variance_ratio)
        @test haskey(flat_est.details, :draws_standard_error)
        @test haskey(flat_est.details, :nested_standard_error)
    end

    @testset "B5 & B12: exact arithmetic on Integer and Bool inputs" begin
        inv = bayes_invert(FiniteBelief([1, 0]), Experiment([1 0; 0 1]))
        @test inv isa BayesianInversion{Rational{Int}}
        @test probabilities(inv.posteriors[1]) == [1 // 1, 0 // 1]
        @test inv.posteriors[2] === nothing

        # Bool payoff/belief/experiment promoted to Int cleanly for certificates.
        bp = DecisionProblem(PayoffTable([true false; false true]),
                             FiniteBelief([true, false]))
        @test eltype(bp.payoff) == Int
        @test eltype(probabilities(bp.belief)) == Int
        json = sprint(io -> export_decision_certificate(io, bp))
        @test !occursin("true", json) && !occursin("false", json)

        # Exact-in / exact-out across formerly Float64-leaking Integer paths.
        p_int = DecisionProblem(PayoffTable([:a, :b], [:s1, :s2], [1 2; 3 4]),
                                FiniteBelief([:s1, :s2], [1, 0]))
        @test eltype(probabilities(restrict_scenarios(p_int, [:s1]).belief)) <: Rational
        @test eltype(aggregate(CVaR(1),
                               DecisionProblem(PayoffTable([1 2; 3 4]),
                                               PointBelief(:s1)))) <: Rational
        @test eltype(relative_regret_table(PayoffTable([1 2; 3 4]))) <: Rational
        @test total_variation([1, 0], [0, 1]) === 1 // 1
        @test_throws ArgumentError total_variation(Float64[], Float64[])
        regs = policy_regions(p_int, PayoffTable([:a, :b], [:s1, :s2], [0 1; 2 0]); from=0,
                              to=2)
        @test all(r -> r.from isa Rational && r.to isa Rational, regs)
        vp = voi_profile(p_int, Experiment([1 0; 0 1]))
        @test vp.evsi isa Rational
        evr_e = expected_variance_reduction(p_int, Experiment([1 0; 0 1]), [10, 20])
        @test evr_e.reduction isa Rational && evr_e.share isa Rational
        evr_p = expected_variance_reduction(p_int, [1, 2], [10, 20])
        @test evr_p.reduction isa Rational && evr_p.share isa Rational
        sp = DecisionProblem(PayoffTable([:none, :act], [:s1, :s2], [0 0; 2 1]),
                             FiniteBelief([:s1, :s2], [1, 0]); baseline=:none)
        sa = survey_or_act(sp, Experiment([1 0; 0 1]); action_costs=[0, 5], survey_cost=1)
        @test sa.act_now.cost_effectiveness isa Rational
        @test sa.survey.cost_effectiveness isa Rational
        am = adaptive_management(p_int, (; a=Experiment([1 0; 0 1])); horizon=2)
        @test eltype(am.policy.belief) <: Rational && am.policy.value isa Rational
    end

    @testset "B6: decision_robustness respects ScenarioSet and PointBelief" begin
        t = PayoffTable([:a, :b], [:s1, :s2, :s3], [10 10 0; 5 5 100])
        ps = DecisionProblem(t, ScenarioSet([:s1, :s2]))
        r = decision_robustness(ps)
        @test first.(r.best) == [:s1, :s2]
        @test r.robust == [:a] && !r.reversal
        pp = DecisionProblem(t, PointBelief(:s3))
        rp = decision_robustness(pp)
        @test first.(rp.best) == [:s3]
        @test rp.robust == [:b] && !rp.reversal
    end

    @testset "B7: tie tolerances across decide, ParetoFront, and Satisfice" begin
        # Integer payoff + Float64 belief: float rounding 0.1 + 0.2 vs 0.3 must tie.
        p_f = DecisionProblem(PayoffTable([:a, :b], [:s1, :s2, :s3], [1 1 0; 0 0 1]),
                              [0.1, 0.2, 0.3] ./ 0.6)
        @test decide(p_f, MaxExpectedUtility()).chosen == [:a, :b]

        # ParetoFront with wildly different objective scales (1e8 vs [0, 1]).
        p_big = DecisionProblem(PayoffTable([:a, :b], [:s1], [1e8; 1e8;;]), [1.0])
        p_small = DecisionProblem(PayoffTable([:a, :b], [:s1], [0.2; 0.8;;]), [1.0])
        pf = decide([p_big, p_small], ParetoFront())
        @test pf.chosen == [:b]

        # Satisfice honors explicit atol.
        psat = DecisionProblem(PayoffTable([:a, :b], [:s1], [9.99; 5.0;;]), [1.0])
        @test decide(psat, Satisfice(10.0); atol=0.05).chosen == [:a]
        @test isempty(decide(psat, Satisfice(10.0); atol=0.0).chosen)
    end

    @testset "B8 & B9: PowerUtility inverse and ExponentialUtility large-scale CE" begin
        @test_throws ArgumentError inverse_utility(PowerUtility(:x; risk_aversion=2.0), 0.0)
        u = ExponentialUtility(:frogs; risk_aversion=0.05)
        # Populations around 1000-1040 saturate -expm1(-50) to 1.0, yet CARA CE must shift cleanly.
        ce_small = certainty_equivalent(u, [0.0, 40.0]; weights=[1, 3])
        ce_large = certainty_equivalent(u, [1000.0, 1040.0]; weights=[1, 3])
        @test isfinite(ce_large)
        @test ce_large ≈ 1000.0 + ce_small atol = 1e-10
        # Also on an OutcomeTable.
        rows = [(; draw=1, action=:a, replicate=1, frogs=1000.0),
                (; draw=1, action=:a, replicate=2, frogs=1040.0)]
        ot = OutcomeTable(rows; outcomes=[:frogs])
        res = only(certainty_equivalent(ot, u))
        @test res.certainty_equivalent ≈ 1000.0 + certainty_equivalent(u, [0.0, 40.0]) atol = 1e-10
    end

    @testset "B10: GAMRegression cross-validation with zero-weight folds" begin
        rng = Xoshiro(77)
        x = randn(rng, 80)
        Q = vcat(zeros(1, 80), permutedims(sin.(x) .+ 0.2 .* randn(rng, 80)))
        prob = DecisionProblem(PayoffTable(Q; actions=[:a, :b]), SampleBelief((; x)))
        cmp = compare_regressions(prob, :x,
                                  (; gam=GAMRegression(), spline=SplineRegression());
                                  folds=4, seed=1)
        @test all(r -> isfinite(r.cv_error) && r.cv_error >= 0, cmp.methods)
    end

    @testset "B11: solve_program does not mutate prog.model" begin
        payoff = PayoffTable([:a, :b], [:s1, :s2], [4 // 1 0; 0 4 // 1])
        credal = CredalSet([:s1, :s2], [[1 // 1, 0], [0, 1 // 1]])
        prob = DecisionProblem(payoff, credal)
        prog = eadmissibility_model(prob, :a)
        @test JuMP.backend(prog.model).state == JuMP.MOIU.NO_OPTIMIZER
        sol1 = solve_program(prog; optimizer=CDDLib.Optimizer{Rational{BigInt}})
        @test sol1.status === :optimal && sol1.exact
        @test JuMP.backend(prog.model).state == JuMP.MOIU.NO_OPTIMIZER
        sol2 = solve_program(prog; optimizer=HiGHS.Optimizer)
        @test sol2.status === :optimal
        @test JuMP.backend(prog.model).state == JuMP.MOIU.NO_OPTIMIZER
    end

    @testset "B13-B22: numerical stability, bounds, screening, and edge guards" begin
        # B13: expected_variance_reduction without catastrophic cancellation at huge offsets.
        base_g = [1e8 + 1.0, 1e8 + 2.0]
        p_big = DecisionProblem(PayoffTable([0.0 0.0; 1.0 2.0]), [0.5, 0.5])
        evr = expected_variance_reduction(p_big, [1, 2], base_g)
        @test evr.prior_variance ≈ 0.25 atol = 1e-12
        @test evr.reduction ≈ 0.25 atol = 1e-12
        @test evr.share ≈ 1.0 atol = 1e-12

        # B14: gaussian_experiment with wide mean spacing (no NaN underflow).
        e_wide = gaussian_experiment([0.0, 1000.0], 0.1; bins=50)
        @test all(isfinite, e_wide.likelihood)
        @test all(isapprox.(sum(e_wide.likelihood; dims=1), 1.0; atol=1e-12))

        # B15: small-shape Beta draws never return NaN.
        rng = Xoshiro(123)
        post = study_posterior(rng, BinomialStudy(:p; n=10), (; x1=0),
                               (; a=0.01, b=0.01); draws=100)
        @test all(isfinite, post.p)
        @test all(x -> 0.0 <= x <= 1.0, post.p)

        # B16: rank_reversals(sweep) ignores ties.
        t1 = PayoffTable([:a, :b], [:s1], [1.0; 1.0;;])
        t2 = PayoffTable([:a, :b], [:s1], [1.0; 2.0;;])
        sw = weight_sweep(w -> DecisionProblem(w == 0 ? t1 : t2, [1.0]), [0, 1])
        @test isempty(rank_reversals(sw))

        # B17: effective_sample_size validation and Rational overflow protection.
        @test_throws ArgumentError effective_sample_size([-0.5, 1.5])
        @test_throws ArgumentError effective_sample_size([0.0, 0.0])
        big_w = [1 // (10^9 + i) for i in 1:5]
        @test isfinite(Float64(effective_sample_size(big_w))) &&
              effective_sample_size(big_w) > 4

        # B18: Inf / empty / domain checks.
        @test_throws ArgumentError PayoffTable([1.0 2.0; 3.0 4.0]) / Inf
        @test_throws ArgumentError PayoffTable([1.0 2.0; 3.0 4.0]) * Inf
        @test_throws ArgumentError NormalStudy(:x; n=10, sd=Inf)
        @test_throws ArgumentError adaptive_management(p_big, (;); horizon=2, discount=Inf)
        @test_throws ArgumentError enbs_curve([0.1, 0.2]; n=[-1, 10])
        @test_throws ArgumentError optimal_sample_size([(; n=0.1, enbs=1.0),
                                                        (; n=0.2, enbs=2.0),
                                                        (; n=0.3, enbs=3.0),
                                                        (; n=0.4, enbs=4.0),
                                                        (; n=0.5, enbs=5.0)]; smooth=true)
        ax = ScenarioAxis(:u, [:lo, :hi]; kind=:uncertainty, weights=[0.5, 0.5])
        des = ScenarioDesign(ax; purpose=:forecasting)
        @test_throws ArgumentError scenario_forecast(des, (; lo=Float64[], hi=[1.0]))

        # B19: weighted screening drops columns constant on positive-weight draws.
        cols_w = (; a=[1.0, 1.0, 1.0, 99.0], b=[1.0, 2.0, 3.0, 4.0])
        kept_cols, dropped = DecisionAnalysis._screen_columns(cols_w,
                                                              [1 / 3, 1 / 3, 1 / 3, 0.0])
        @test keys(kept_cols) == (:b,) && dropped == [:a]

        # B22: PointBelief in probability_optimal and colliding conditional_problem labels.
        po = probability_optimal(DecisionProblem(PayoffTable([:a, :b], [:s1, :s2],
                                                             [1 5; 3 2]),
                                                 PointBelief(:s2)))
        @test [(r.action, r.probability) for r in po] == [(:a, 1), (:b, 0)]
        col_b = SampleBelief((; x=["a_b", "a"], y=["c", "b_c"]))
        col_p = DecisionProblem(PayoffTable([1.0 2.0; 3.0 4.0]; actions=[:a, :b]), col_b)
        cp = conditional_problem(col_p, (:x, :y))
        @test length(state_labels(cp)) == 2 && allunique(state_labels(cp))

        # B25: select_under_budget passes count when nothing fits in pass 1, and show huge Rational.
        units = [(; choice=:act_now, value=10, cost=50, largest_cost=50)]
        @test select_under_budget(units, 10).passes == 1
        @test isfinite(equivalent_binomial_sample_size(0.5, sqrt(0.25 / (1 + 1e-12))))
        @test equivalent_binomial_sample_size(0.5, sqrt(0.25 / (1 + 1e-12))) > 0
        huge_r = (big(10)^400 + 1) // big(10)^400
        @test !occursin("NaN", DecisionAnalysis._cell(huge_r))
        @test occursin("\\|", DecisionAnalysis._md_code("a|b"))
    end

    @testset "Track 2: performance optimizations and mathematical parity" begin
        # O1: GaussianApproximation vector n matches scalar n (including bootstrap).
        rng = Xoshiro(4242)
        n = 500
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        Q = vcat(zeros(1, n), permutedims(1.0 .+ x1 .+ 2.0 .* x2))
        prob = DecisionProblem(PayoffTable(Q; actions=[:a, :b]), SampleBelief((; x1, x2)))
        vec_est = evsi(prob, GaussianApproximation(:x1; n0=5, n=[10, 50, 200]);
                       bootstrap=25, seed=7)
        for (i, ni) in enumerate([10, 50, 200])
            sc = evsi(prob, GaussianApproximation(:x1; n0=5, n=ni); bootstrap=25, seed=7)
            @test vec_est[i].value ≈ sc.value rtol = 1e-12 atol = 1e-12
            @test vec_est[i].standard_error ≈ sc.standard_error rtol = 1e-12 atol = 1e-12
        end

        # O3: GP analytical gradient matches central finite differences to 1e-7.
        n_gp, p_gp = 45, 3
        X = randn(rng, n_gp, p_gp)
        H = hcat(ones(n_gp), X)
        y = 0.5 .+ X * [1.2, -0.7, 0.4] .+ 0.3 .* sin.(X[:, 1] .* X[:, 2]) .+
            0.2 .* randn(rng, n_gp)
        noise = 0.5 .+ rand(rng, n_gp)
        lo = fill(-5.0, p_gp + 1)
        hi = fill(5.0, p_gp + 1)
        C = Matrix{Float64}(undef, n_gp, n_gp)
        A = Matrix{Float64}(undef, n_gp, n_gp)
        P = Matrix{Float64}(undef, n_gp, n_gp)
        Xs = Matrix{Float64}(undef, n_gp, p_gp)
        norms = Vector{Float64}(undef, n_gp)
        MX = Matrix{Float64}(undef, n_gp, p_gp)
        msum = Vector{Float64}(undef, n_gp)
        θ = [0.3, -0.2, 0.7, -1.5]
        g_ana = zeros(p_gp + 1)
        ll0 = DecisionAnalysis._gp_ll_and_grad!(g_ana, X, H, y, noise, θ, lo, hi, C, A, P,
                                                Xs, norms, MX, msum)
        @test isfinite(ll0)
        ε = 1e-6
        g_fd = zeros(p_gp + 1)
        for k in eachindex(θ)
            θp = copy(θ)
            θp[k] += ε
            θm = copy(θ)
            θm[k] -= ε
            lp = DecisionAnalysis._gp_ll_and_grad!(nothing, X, H, y, noise, θp, lo, hi, C,
                                                   A, P, Xs, norms, MX, msum)
            lm = DecisionAnalysis._gp_ll_and_grad!(nothing, X, H, y, noise, θm, lo, hi, C,
                                                   A, P, Xs, norms, MX, msum)
            g_fd[k] = (lp - lm) / (2 * ε)
        end
        @test g_ana ≈ g_fd rtol = 1e-6 atol = 1e-7

        # O4: outcomes threaded=true matches threaded=false bit-for-bit, and hoists verbs.
        wp_c, aa_c = Ref(0), Ref(0)
        cm = _CountModel(0.0, wp_c, aa_c)
        draws_c = (; x=collect(1.0:8.0))
        acts_c = (; a=0.0, b=1.0, c=2.0)
        ot_serial = outcomes(cm, acts_c, draws_c; replicates=5, seed=11, threaded=false)
        @test wp_c[] == 8
        @test aa_c[] == 8 * 3
        ot_thread = outcomes(_CountModel(0.0, Ref(0), Ref(0)), acts_c, draws_c;
                             replicates=5, seed=11, threaded=true)
        @test ot_thread.values == ot_serial.values
        @test ot_thread.status == ot_serial.status
        # Fast-path single-attribute utilities match FunctionUtility.
        for u in (LinearUtility(:y; worst=0, best=10),
                  ExponentialUtility(:y; risk_aversion=0.1, worst=0, best=10),
                  PowerUtility(:y; risk_aversion=0.5, worst=0, best=20))
            pt1 = payoff_table(ot_serial, u)
            pt2 = payoff_table(ot_serial, FunctionUtility(o -> u(o)))
            @test pt1.Q ≈ pt2.Q rtol = 1e-12 atol = 1e-12
            @test pt1.standard_errors ≈ pt2.standard_errors rtol = 1e-12 atol = 1e-12
        end

        # O6: _binomial (BINV and BTPE) exact edges and moments at large n.
        brng = Xoshiro(2026)
        @test DecisionAnalysis._binomial(brng, 10_000, 0.0) == 0
        @test DecisionAnalysis._binomial(brng, 10_000, 1.0) == 10_000
        for (nb, pb) in ((20, 0.05), (50, 0.95), (1000, 0.005), (10_000, 0.3),
                         (10_000, 0.7), (10_000, 0.5))
            draws_b = [DecisionAnalysis._binomial(brng, nb, pb) for _ in 1:4000]
            @test all(x -> 0 <= x <= nb, draws_b)
            μ = nb * pb
            σ2 = nb * pb * (1 - pb)
            m_hat = sum(draws_b) / length(draws_b)
            v_hat = sum(x -> (x - m_hat)^2, draws_b) / (length(draws_b) - 1)
            @test abs(m_hat - μ) < 4 * sqrt(σ2 / length(draws_b))
            @test abs(v_hat - σ2) / σ2 < 0.1
        end

        # O8: _solve_internal lower-bound shift handles negative, positive, and fixed bounds.
        m_lp = JuMP.GenericModel{Rational{BigInt}}()
        JuMP.@variable(m_lp, -3 // 1 <= x1 <= 2 // 1)
        JuMP.@variable(m_lp, 4 // 1 <= x2 <= 10 // 1)
        JuMP.@variable(m_lp, 5 // 1 <= x3 <= 5 // 1)
        JuMP.@constraint(m_lp, x1 + x2 + x3 >= 10 // 1)
        JuMP.@objective(m_lp, Min, 2 * x1 + 3 * x2 + x3)
        data_lp = JuMP.lp_matrix_data(m_lp)
        st, xv = DecisionAnalysis._solve_internal(data_lp)
        @test st === :optimal
        @test xv == Rational{BigInt}[1 // 1, 4 // 1, 5 // 1]
    end
end
