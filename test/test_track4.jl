@testset "Track 4a: horizontal ecosystem bridges" begin
    @testset "T4a.3 FiniteKernels bidirectional & Markov transitions" begin
        b = FiniteBelief([:absent, :present], [3 // 5, 2 // 5])
        kb = FiniteKernels.FiniteKernel(b; name=:Disease)
        @test kb isa FiniteKernels.FiniteKernel{Rational{Int},1}
        @test FiniteBelief(kb) == b

        e = Experiment([4 // 5 1 // 5; 1 // 5 4 // 5];
                       signals=[:neg, :pos], states=[:absent, :present])
        ke = FiniteKernels.FiniteKernel(e; state_name=:Disease, signal_name=:Swab)
        @test ke isa FiniteKernels.FiniteKernel{Rational{Int},2}
        @test Experiment(ke).likelihood == e.likelihood
        @test state_labels(Experiment(ke)) == state_labels(e)
        @test signal_labels(Experiment(ke)) == signal_labels(e)

        # Unlabelled experiment gets default state labels when lifted to a kernel.
        eu = Experiment([0.7 0.2; 0.3 0.8]; signals=[:low, :high])
        ku = FiniteKernels.FiniteKernel(eu)
        @test FiniteKernels.kernel_matrix(ku) == eu.likelihood

        # FiniteKernel and ResilienceCore.MarkovKernel as transitions in adaptive_management.
        states = [:absent, :present]
        actions = [:wait, :treat]
        payoff = PayoffTable(actions, states, [10 // 1 0 // 1; 7 // 1 6 // 1])
        prob = DecisionProblem(payoff, FiniteBelief(states, [1 // 2, 1 // 2]);
                               baseline=:wait)
        obs = Experiment([9 // 10 2 // 10; 1 // 10 8 // 10];
                         signals=[:neg, :pos], states)

        T_wait = [9 // 10 1 // 10; 1 // 10 9 // 10]
        T_treat = [9 // 10 7 // 10; 1 // 10 3 // 10]
        ax = FiniteKernels.FiniteSpace(FiniteKernels.FiniteAxis(:State, states))
        k_wait = FiniteKernels.FiniteKernel(ax, ax, T_wait; atol=0)
        k_treat = FiniteKernels.FiniteKernel(ax, ax, T_treat; atol=0)

        # Single FiniteKernel for all actions (exact rational arithmetic preserved).
        r_mat = adaptive_management(prob, obs; transitions=T_wait, horizon=3)
        r_fk = adaptive_management(prob, obs; transitions=k_wait, horizon=3)
        @test r_fk.active == r_mat.active
        @test r_fk.active isa Rational

        # Per-action NamedTuple of FiniteKernels.
        r_named_mat = adaptive_management(prob, obs;
                                          transitions=(; wait=T_wait, treat=T_treat),
                                          horizon=3)
        r_named_fk = adaptive_management(prob, obs;
                                         transitions=(; wait=k_wait, treat=k_treat),
                                         horizon=3)
        @test r_named_fk.active == r_named_mat.active
        @test r_named_fk.passive == r_named_mat.passive

        # Time-varying schedule of FiniteKernels.
        r_tv_mat = adaptive_management(prob, obs; transitions=[T_wait, T_treat, T_wait],
                                       horizon=3)
        r_tv_fk = adaptive_management(prob, obs; transitions=[k_wait, k_treat, k_wait],
                                      horizon=3)
        @test r_tv_fk.active == r_tv_mat.active

        # Label mismatch in FiniteKernel transition throws LabelMismatchError.
        wrong_ax = FiniteKernels.FiniteSpace(FiniteKernels.FiniteAxis(:State, [:a, :b]))
        k_wrong = FiniteKernels.FiniteKernel(wrong_ax, wrong_ax, T_wait; atol=0)
        @test_throws LabelMismatchError adaptive_management(prob, obs; transitions=k_wrong,
                                                            horizon=2)

        # ResilienceCore.MarkovKernel (row-stochastic -> column-stochastic automatically).
        mk_wait = FiniteKernels.ResilienceCore.linearise(k_wait)
        mk_treat = FiniteKernels.ResilienceCore.linearise(k_treat)
        r_mk = adaptive_management(prob, obs;
                                   transitions=(; wait=mk_wait, treat=mk_treat),
                                   horizon=3)
        @test r_mk.active ≈ Float64(r_named_mat.active) atol = 1e-10
        r_mk_single = adaptive_management(prob, obs; transitions=mk_wait, horizon=3)
        @test r_mk_single.active ≈ Float64(r_mat.active) atol = 1e-10

        # Direct FiniteKernel Bayesian inversion (delegates to FiniteKernels.invert_kernel when available).
        inv_fk = bayes_invert(kb, ke)
        inv_da = bayes_invert(b, e)
        @test inv_fk isa BayesianInversion{Rational{Int}}
        @test inv_fk.signals == inv_da.signals
        @test inv_fk.states == inv_da.states
        @test inv_fk.joint == inv_da.joint
        @test inv_fk.marginal == inv_da.marginal
        @test inv_fk.posteriors == inv_da.posteriors

        # Zero-evidence signal yields `nothing` posterior without error.
        e_zero = Experiment([1 // 1 1 // 1; 0 // 1 0 // 1];
                            signals=[:always, :never], states=[:absent, :present])
        ke_zero = FiniteKernels.FiniteKernel(e_zero; state_name=:Disease, signal_name=:Swab)
        inv_zero = bayes_invert(kb, ke_zero)
        @test inv_zero.marginal == [1 // 1, 0 // 1]
        @test inv_zero.posteriors[1] == b
        @test inv_zero.posteriors[2] === nothing
    end

    @testset "T4a.4 GAMRegression bam/scam/ti, diagnostics, and parameter_draws" begin
        rng = Xoshiro(202641)
        n = 600
        x1 = randn(rng, n)
        x2 = randn(rng, n)
        # Payoff difference is monotone increasing in x1 plus smooth interaction in (x1, x2).
        diff_true = 1.5 .* x1 .+ 0.4 .* sin.(x2) .+ 0.3 .* x1 .* x2
        Q = vcat(zeros(1, n), permutedims(diff_true .+ 0.5 .* randn(rng, n)))
        prob = DecisionProblem(PayoffTable(Q; actions=[:ref, :act]),
                               SampleBelief((; x1, x2)); baseline=:ref)

        # Standard GAM surfaces edf and k_check diagnostics in details.
        g_std = evppi(prob, :x1, GAMRegression())
        @test g_std.value > 0
        @test haskey(g_std.details, :edf)
        @test haskey(g_std.details, :k_check)
        @test length(g_std.details.edf) == 2
        @test g_std.details.backend === :gam

        # GAM.bam (default smoothing=:REML, both dense and discrete=true).
        g_bam = evppi(prob, :x1, GAMRegression(; backend=:bam))
        g_bam_disc = evppi(prob, :x1, GAMRegression(; backend=:bam, discrete=true))
        @test g_bam.details.backend === :bam
        @test g_bam.details.smoothing === :REML
        @test isapprox(g_bam.value, g_std.value; rtol=0.15)
        @test isapprox(g_bam_disc.value, g_bam.value; rtol=0.15)

        # GAM.scam with monotone increasing shape constraint (:mpi).
        g_scam = evppi(prob, :x1, GAMRegression(; shape=:mpi))
        @test g_scam.details.backend === :scam
        @test isapprox(g_scam.value, g_std.value; rtol=0.15)

        # Pure interaction decomposition with ti(...) vs te(...).
        g_te = evppi(prob, (:x1, :x2),
                     GAMRegression(; tensor=true, interaction=:te, smoothing=:REML))
        g_ti = evppi(prob, (:x1, :x2),
                     GAMRegression(; tensor=true, interaction=:ti, smoothing=:REML))
        @test isapprox(g_ti.value, g_te.value; rtol=0.15)
        @test g_ti.value > g_std.value

        # Constructor validation.
        @test_throws ArgumentError GAMRegression(; backend=:unknown)
        @test_throws ArgumentError GAMRegression(; interaction=:bad)
        @test_throws ArgumentError GAMRegression(; shape=:bad)
        @test_throws ArgumentError GAMRegression(; backend=:bam, smoothing=:GCV)
        @test_throws ArgumentError GAMRegression(; backend=:bam, shape=:mpi)

        # parameter_draws & SampleBelief from a fitted GAM.GamModel.
        df = (; y=2.0 .* x1 .- 0.5 .* x1 .^ 2 .+ 0.3 .* randn(rng, n), x1)
        m = GAM.gam(GAM.@formula(y ~ s(x1, k=6)), df)
        draws = parameter_draws(m; n=250, seed=42)
        @test draws isa NamedTuple
        @test length(first(draws)) == 250
        @test draws == parameter_draws(m; n=250, seed=42)
        sb = SampleBelief(m; n=250, seed=42)
        @test sb isa SampleBelief
        @test length(sb.weights) == 250
        @test_throws ArgumentError parameter_draws(m; n=0)
    end
end
