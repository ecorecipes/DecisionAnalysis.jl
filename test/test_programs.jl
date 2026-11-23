# The linear programmes as JuMP models, and the three routes to solve one model: the
# built-in exact simplex fed from the matrix form, an exact optimiser (CDDLib in
# `Rational{BigInt}`), and a floating-point optimiser (HiGHS) with rounding and exact
# verification. The oracles here are written from the definitions and use nothing of
# the package but its public results.
using JuMP: JuMP
using CDDLib: CDDLib
using HiGHS: HiGHS

const CDD = CDDLib.Optimizer{BigRat}
const HIGHS = HiGHS.Optimizer

# Outcomes of the floating-point route over the whole file: verified exactly, or
# refused with the typed error. Nothing else is allowed.
const FLOAT_ROUTE = Dict(:verified => 0, :refused => 0)

# Run `f` on the floating-point route; `nothing` if it refused with the typed error.
function float_route(f)
    try
        out = f()
        FLOAT_ROUTE[:verified] += 1
        return out
    catch err
        err isa ExactVerificationError || rethrow()
        FLOAT_ROUTE[:refused] += 1
        return nothing
    end
end

function affine_constraints(model)
    return JuMP.num_constraints(model; count_variable_in_set_constraints=false)
end
function bound_constraints(model)
    return JuMP.num_constraints(model; count_variable_in_set_constraints=true) -
           affine_constraints(model)
end

# Oracle: `weights` place a prior in the hull under which action `a` is optimal.
function oracle_is_eadmissible_witness(Q, points, weights, a)
    all(w -> w >= 0, weights) || return false
    sum(weights) == 1 || return false
    prior = [sum(weights[k] * points[k][t] for k in eachindex(points)) for t in axes(Q, 2)]
    eu = oracle_expected_utility(Q, prior)
    return all(b -> eu[b] <= eu[a], eachindex(eu))
end

# Oracle: `G` is column-stochastic and reproduces the garbled likelihood.
function oracle_is_garbling_witness(G, L, Lt)
    all(x -> x >= 0, G) || return false
    all(z -> sum(G[zt, z] for zt in axes(G, 1)) == 1, axes(G, 2)) || return false
    return all(sum(G[zt, z] * L[z, t] for z in axes(L, 1)) == Lt[zt, t]
               for zt in axes(Lt, 1), t in axes(L, 2))
end

# Oracle for the largest total-variation distance, over states, between G·L and L′.
function oracle_garbled_distance(G, L, Lt)
    return maximum(sum(abs(sum(G[zt, z] * L[z, t] for z in axes(L, 1)) - Lt[zt, t])
                       for zt in axes(Lt, 1)) / 2 for t in axes(L, 2))
end

@testset "linear programmes through JuMP" begin
    @testset "JuMP solves a rational model exactly with CDDLib" begin
        # The scratch check of the design note: JuMP 1.32 with CDDLib 0.10.
        model = JuMP.GenericModel{BigRat}(CDD)
        JuMP.@variable(model, x >= 0)
        JuMP.@variable(model, y >= 0)
        JuMP.@constraint(model, x + 3y <= 6)
        JuMP.@constraint(model, 3x + y <= 6)
        JuMP.@objective(model, Max, x + y)
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == JuMP.MOI.OPTIMAL
        @test JuMP.value(x) isa BigRat
        @test (JuMP.value(x), JuMP.value(y)) == (3 // 2, 3 // 2)
        @test JuMP.objective_value(model) == 3
        # The matrix form is available in rationals; infinite bounds are ±1//0.
        data = JuMP.lp_matrix_data(model)
        @test data isa JuMP.LPMatrixData{BigRat}
        @test Matrix(data.A) == [1 3; 3 1]
        @test data.b_upper == [6, 6] && all(isinf, data.b_lower)
        @test data.x_lower == [0, 0] && all(isinf, data.x_upper)
    end

    @testset "documented shapes" begin
        rng = Random.Xoshiro(20261124)
        for _ in 1:20
            m, n, K = rand(rng, 1:5), rand(rng, 2:4), rand(rng, 1:4)
            problem = DecisionProblem(random_payoff_table(rng, m, n; T=BigRat),
                                      random_credal_set(rng, n, K; T=BigRat))
            prog = eadmissibility_model(problem, rand(rng, feasible_actions(problem)))
            @test prog isa DecisionProgram{BigRat}
            @test prog.name === :e_admissibility
            @test prog.model isa JuMP.GenericModel{BigRat}
            @test JuMP.num_variables(prog.model) == K + 1
            @test affine_constraints(prog.model) == m + 1
            @test bound_constraints(prog.model) == K
            @test size(prog.variables.λ) == (K,)
            @test length(prog.constraints.optimal) == m
            @test JuMP.objective_sense(prog.model) == JuMP.MOI.MAX_SENSE

            env = evpi_envelope_model(problem)
            @test env.name === :evpi_envelope
            @test JuMP.num_variables(env.model) == K + 1
            @test affine_constraints(env.model) == m + 1
            @test bound_constraints(env.model) == K
            @test length(env.constraints.act_now) == m

            nz, nt, S = rand(rng, 2:4), rand(rng, 2:4), rand(rng, 2:3)
            e = random_experiment(rng, nz, S; T=BigRat)
            f = random_experiment(rng, nt, S; T=BigRat)
            gar = garbling_model(f, e)
            @test gar.name === :garbling
            @test JuMP.num_variables(gar.model) == nt * nz
            @test affine_constraints(gar.model) == nz + nt * S
            @test bound_constraints(gar.model) == nt * nz
            @test size(gar.variables.G) == (nt, nz)
            @test JuMP.objective_sense(gar.model) == JuMP.MOI.FEASIBILITY_SENSE

            def = deficiency_model(e, f)
            @test def.name === :deficiency
            @test JuMP.num_variables(def.model) == nt * nz + nt * S + 1
            @test affine_constraints(def.model) == nz + 2 * nt * S + S
            @test bound_constraints(def.model) == nt * nz + nt * S + 1
            @test size(def.variables.u) == (nt, S)
            @test JuMP.objective_sense(def.model) == JuMP.MOI.MIN_SENSE
        end
        # A floating-point problem gives a floating-point model.
        fp = DecisionProblem(PayoffTable([1.0 0.0; 0.0 1.0]),
                             CredalSet([[0.5, 0.5], [0.25, 0.75]]))
        @test eadmissibility_model(fp, :a1).model isa JuMP.GenericModel{Float64}
        @test_throws ArgumentError eadmissibility_model(fp, :zzz)
        # Integer and Rational{Int} problems are stated in Rational{BigInt}.
        ip = DecisionProblem(PayoffTable([1 0; 0 1]),
                             CredalSet([[1 // 2, 1 // 2], [1 // 4, 3 // 4]]))
        @test eadmissibility_model(ip, :a1) isa DecisionProgram{BigRat}
    end

    @testset "matrix form to the built-in simplex: bounds and ranges" begin
        # General bounds and two-sided rows, which the package's own programmes do
        # not use, against CDDLib on the same model.
        rng = Random.Xoshiro(20261125)
        for _ in 1:40
            n, m = rand(rng, 2:4), rand(rng, 1:3)
            model = JuMP.GenericModel{BigRat}()
            lo = BigRat.(rand(rng, -3:1, n))
            x = JuMP.@variable(model, [j = 1:n], lower_bound = lo[j],
                               upper_bound = lo[j] + rand(rng, 1:5))
            free = JuMP.@variable(model)
            A = BigRat.(rand(rng, -3:3, m, n))
            centre = A * (lo .+ 1 // 2)
            JuMP.@constraint(model, [i = 1:m],
                             centre[i] - 2 <= sum(A[i, j] * x[j] for j in 1:n) + free <=
                             centre[i] + 3)
            JuMP.@constraint(model, free >= -2)
            JuMP.@constraint(model, free <= 4 + sum(x))
            c = BigRat.(rand(rng, -4:4, n))
            sense = rand(rng, (JuMP.MOI.MIN_SENSE, JuMP.MOI.MAX_SENSE))
            JuMP.@objective(model, sense, sum(c[j] * x[j] for j in 1:n) + 2free + 1 // 3)
            prog = DecisionProgram(:test, model, (; x, free), (;))
            internal = solve_program(prog)
            exact = solve_program(prog; optimizer=CDD)
            @test internal.status === :optimal && exact.status === :optimal
            @test internal.objective == exact.objective
            @test internal.exact && exact.exact
            @test (internal.route, exact.route) == (:internal, :exact_optimizer)
            @test size(internal.values.x) == (n,)
            float = solve_program(prog; optimizer=HIGHS)
            @test float.route === :float_optimizer
            float.exact && @test float.objective == exact.objective
        end
        # Infeasible and unbounded models are reported, not thrown.
        model = JuMP.GenericModel{BigRat}()
        x = JuMP.@variable(model, lower_bound = 0)
        JuMP.@constraint(model, x <= -1)
        prog = DecisionProgram(:test, model, (; x), (;))
        for opt in (nothing, CDD, HIGHS)
            sol = solve_program(prog; optimizer=opt)
            @test sol.status === :infeasible
            @test sol.objective === nothing && isempty(sol.values)
            @test sol.exact == (opt !== HIGHS)
        end
        model = JuMP.GenericModel{BigRat}()
        x = JuMP.@variable(model, lower_bound = 0)
        JuMP.@objective(model, Max, x)
        @test solve_program(DecisionProgram(:test, model, (; x), (;))).status === :unbounded
        @test_throws ArgumentError solve_program(prog; rounding_tol=-1)
        # An exact optimiser cannot take a floating-point model, nor the reverse
        # without rounding; a rational-only optimiser on a float model is refused.
        fmodel = JuMP.GenericModel{Float64}()
        y = JuMP.@variable(fmodel, lower_bound = 0.0)
        JuMP.@objective(fmodel, Min, y)
        fprog = DecisionProgram(:test, fmodel, (; y), (;))
        @test_throws ArgumentError solve_program(fprog; optimizer=CDD)
        sol = solve_program(fprog; optimizer=HIGHS)
        @test sol.status === :optimal && sol.objective == 0.0 && !sol.exact
        @test sol.route === :float_optimizer
        sol = solve_program(fprog)
        @test sol.objective === 0.0 && !sol.exact && sol.route === :internal
    end

    @testset "E-admissibility: three routes agree" begin
        rng = Random.Xoshiro(20261126)
        for _ in 1:60
            m, n, K = rand(rng, 2:5), rand(rng, 2:4), rand(rng, 1:4)
            problem = DecisionProblem(random_payoff_table(rng, m, n; T=BigRat),
                                      random_credal_set(rng, n, K; T=BigRat))
            Q, points = problem.payoff.Q, extreme_points(problem.belief)
            internal = decide(problem, EAdmissible())
            exact = decide(problem, EAdmissible(); optimizer=CDD)
            @test exact.chosen == internal.chosen
            @test internal.details.route === :internal
            @test exact.details.route === :exact_optimizer
            for r in (internal, exact)
                @test all(values(r.details.exact))
                @test Set(keys(r.details.exact)) == Set(r.actions)
                for a in r.chosen
                    i = findfirst(==(a), r.actions)
                    @test oracle_is_eadmissible_witness(Q, points, r.details.weights[a], i)
                end
            end
            float = float_route(() -> decide(problem, EAdmissible(); optimizer=HIGHS))
            float === nothing && continue
            @test float.chosen == internal.chosen
            @test float.details.route === :float_optimizer
            for a in float.chosen
                i = findfirst(==(a), float.actions)
                @test float.details.exact[a]
                @test eltype(float.details.weights[a]) == BigRat
                @test oracle_is_eadmissible_witness(Q, points, float.details.weights[a], i)
            end
            # Each programme has the same optimum by every route that certifies it.
            for a in feasible_actions(problem)
                prog() = eadmissibility_model(problem, a)
                ref = solve_program(prog())
                @test solve_program(prog(); optimizer=CDD).objective == ref.objective
                fl = solve_program(prog(); optimizer=HIGHS)
                fl.exact && @test fl.objective == ref.objective
                @test ref.objective <= 0
                @test (ref.objective == 0) == (a in internal.chosen)
            end
        end
    end

    @testset "E-admissibility at a tie: optimal only at one interior prior" begin
        # :hedge is optimal only at the midpoint, where all three actions tie, so a
        # rounded floating-point witness must be exactly the midpoint to pass.
        payoff = PayoffTable([:left, :right, :hedge], [:s1, :s2],
                             BigRat[4 0; 0 4; 2 2])
        problem = DecisionProblem(payoff, CredalSet([:s1, :s2], [[1 // 1, 0], [0, 1 // 1]]))
        for opt in (nothing, CDD)
            r = decide(problem, EAdmissible(); optimizer=opt)
            @test r.chosen == [:left, :right, :hedge]
            @test r.details.weights[:hedge] == [1 // 2, 1 // 2]
        end
        r = float_route(() -> decide(problem, EAdmissible(); optimizer=HIGHS))
        if r !== nothing
            @test r.chosen == [:left, :right, :hedge]
            @test r.details.weights[:hedge] == [1 // 2, 1 // 2]
        end
        # A floating-point problem by a floating-point optimiser is numerical.
        fp = DecisionProblem(PayoffTable([:left, :right, :hedge], [:s1, :s2],
                                         [4.0 0.0; 0.0 4.0; 2.5 2.5]),
                             CredalSet([:s1, :s2], [[1.0, 0.0], [0.0, 1.0]]))
        r = decide(fp, EAdmissible(); optimizer=HIGHS)
        @test r.chosen == [:left, :right, :hedge]
        @test !any(values(r.details.exact))
        @test r.details.weights[:hedge] isa Vector{Float64}
        @test decide(fp, EAdmissible()).chosen == r.chosen
        @test !any(values(decide(fp, EAdmissible()).details.exact))
        # An exact problem with a tolerance is admitted to tolerance, not exactly.
        loose = decide(problem, EAdmissible(); atol=1 // 10)
        @test !any(values(loose.details.exact))
    end

    @testset "garbling: three routes agree" begin
        rng = Random.Xoshiro(20261127)
        certified_no, all_no = 0, 0
        yes = 0
        for trial in 1:60
            nz, S = rand(rng, 2:4), rand(rng, 2:3)
            e = random_experiment(rng, nz, S; T=BigRat)
            nt = rand(rng, 2:4)
            f = isodd(trial) ? garble(e, random_stochastic(rng, BigRat, nt, nz)) :
                random_experiment(rng, nt, S; T=BigRat)
            L, Lt = e.likelihood, f.likelihood
            internal = garbling_result(f, e)
            exact = garbling_result(f, e; optimizer=CDD)
            @test internal.is_garbling == exact.is_garbling
            isodd(trial) && @test internal.is_garbling
            yes += internal.is_garbling
            @test internal.exact && exact.exact
            @test (internal.route, exact.route) == (:internal, :exact_optimizer)
            for r in (internal, exact)
                if r.is_garbling
                    @test r.witness isa Matrix{BigRat}
                    @test oracle_is_garbling_witness(r.witness, L, Lt)
                else
                    @test r.witness === nothing
                end
            end
            @test (is_garbling_of(f, e) !== nothing) == internal.is_garbling
            @test (is_garbling_of(f, e; optimizer=CDD) !== nothing) == internal.is_garbling
            # The feasibility programme and the deficiency programme say the same.
            @test internal.is_garbling == iszero(deficiency(e, f))
            float = float_route(() -> garbling_result(f, e; optimizer=HIGHS))
            float === nothing && continue
            @test float.is_garbling == internal.is_garbling
            @test float.route === :float_optimizer
            if float.is_garbling
                @test float.exact
                @test oracle_is_garbling_witness(float.witness, L, Lt)
            else
                # A "no" from a floating-point optimiser is exact only when a separating
                # payoff table passed the exact check; then that table can be exported
                # and stands up to an independent check.
                @test float.witness === nothing
                if float.exact
                    check_not_garbling_witness(sprint(io -> export_not_garbling_witness(io,
                                                                                        f,
                                                                                        e;
                                                                                        optimizer=HIGHS)))
                end
                certified_no += float.exact
                all_no += 1
            end
        end
        @test 30 <= yes < 60
        # Most "no" answers of the floating-point route are certified on these problems.
        @test certified_no >= (9 * all_no) ÷ 10
        # Floating-point likelihoods: numerical by every route.
        e = Experiment([0.8 0.2; 0.2 0.8])
        noisy = garble(e, [0.75 0.25; 0.25 0.75])
        for opt in (nothing, HIGHS)
            r = garbling_result(noisy, e; optimizer=opt)
            @test r.is_garbling && !r.exact
            @test r.witness * e.likelihood ≈ noisy.likelihood atol = 1e-7
            @test !garbling_result(e, noisy; optimizer=opt).is_garbling
        end
    end

    @testset "deficiency: three routes agree" begin
        rng = Random.Xoshiro(20261128)
        for trial in 1:40
            nz, nt, S = rand(rng, 2:3), rand(rng, 2:3), rand(rng, 2:3)
            e = random_experiment(rng, nz, S; T=BigRat)
            f = trial % 4 == 0 ? garble(e, random_stochastic(rng, BigRat, nt, nz)) :
                random_experiment(rng, nt, S; T=BigRat)
            d = deficiency(e, f)
            @test d isa BigRat
            @test deficiency(e, f; optimizer=CDD) == d
            fl = float_route(() -> deficiency(e, f; optimizer=HIGHS))
            fl === nothing || @test fl == d
            # The minimising garbling attains the value, and no column-stochastic
            # matrix does better (a sample of them, and the witness of each route).
            for opt in (nothing, CDD)
                sol = solve_program(deficiency_model(e, f); optimizer=opt)
                @test sol.objective == 2d
                @test oracle_garbled_distance(sol.values.G, e.likelihood, f.likelihood) == d
            end
            for _ in 1:5
                G = random_stochastic(rng, BigRat, nt, nz)
                @test oracle_garbled_distance(G, e.likelihood, f.likelihood) >= d
            end
        end
        # Floating point by a floating-point optimiser.
        e = Experiment([0.8 0.2; 0.2 0.8])
        noisy = garble(e, [0.75 0.25; 0.25 0.75])
        @test deficiency(noisy, e; optimizer=HIGHS) ≈ 0.15 atol = 1e-7
        @test deficiency(e, noisy; optimizer=HIGHS) ≈ 0 atol = 1e-7
    end

    @testset "EVPI envelope: three routes agree" begin
        rng = Random.Xoshiro(20261129)
        for _ in 1:60
            m, n, K = rand(rng, 2:5), rand(rng, 2:4), rand(rng, 1:4)
            problem = DecisionProblem(random_payoff_table(rng, m, n; T=BigRat),
                                      random_credal_set(rng, n, K; T=BigRat))
            v = robust_evpi(problem)
            @test robust_evpi(problem; optimizer=CDD).upper == v.upper
            fl = float_route(() -> robust_evpi(problem; optimizer=HIGHS))
            fl === nothing || @test fl.upper == v.upper
            # The optimum is the single-prior EVPI at the mixture found, and no
            # prior on a grid over the hull does better.
            sol = solve_program(evpi_envelope_model(problem))
            @test sol.objective == v.upper
            points = extreme_points(problem.belief)
            mix(w) = [sum(w[k] * points[k][t] for k in 1:K) for t in 1:n]
            @test oracle_evpi(problem.payoff.Q, mix(sol.values.λ)) == v.upper
            for w in simplex_grid(K, 6)
                @test oracle_evpi(problem.payoff.Q, mix(w)) <= v.upper
            end
        end
    end

    @testset "exporters by every route, and the solver is never trusted" begin
        rng = Random.Xoshiro(20261130)
        for _ in 1:15
            m, n, K = rand(rng, 2:4), rand(rng, 2:3), rand(rng, 1:3)
            problem = DecisionProblem(random_payoff_table(rng, m, n; T=BigRat),
                                      random_credal_set(rng, n, K; T=BigRat))
            chosen = decide(problem, EAdmissible()).chosen
            for a in feasible_actions(problem), opt in (CDD, HIGHS)
                if a in chosen
                    text = float_route(() -> sprint((io, p, x) -> export_eadmissible_witness(io,
                                                                                             p,
                                                                                             x;
                                                                                             optimizer=opt),
                                                    problem, a))
                    text === nothing || check_eadmissible_witness(text)
                else
                    @test_throws WitnessError export_eadmissible_witness(IOBuffer(),
                                                                         problem, a;
                                                                         optimizer=opt)
                end
            end
            e = random_experiment(rng, rand(rng, 2:3), n; T=BigRat)
            f = garble(e,
                       random_stochastic(rng, BigRat, rand(rng, 2:3),
                                         size(e.likelihood, 1)))
            for opt in (CDD, HIGHS)
                text = float_route(() -> sprint((io, x, y) -> export_garbling_witness(io, x,
                                                                                      y;
                                                                                      optimizer=opt),
                                                f, e))
                text === nothing || check_garbling_witness(text)
            end
        end
        # A rounding tolerance so coarse that the answer cannot survive it is refused
        # with the typed error; nothing unverified comes back.
        e = Experiment(BigRat[97//100 1//100; 3//100 99//100])
        f = garble(e, BigRat[89//97 1//89; 8//97 88//89])
        @test deficiency(e, f) == 0
        @test_throws ExactVerificationError deficiency(f, e; optimizer=HIGHS,
                                                       rounding_tol=0.05)
        @test_throws ExactVerificationError garbling_result(f, e; optimizer=HIGHS,
                                                            rounding_tol=0.05)
        err = try
            deficiency(f, e; optimizer=HIGHS, rounding_tol=0.05)
        catch caught
            caught
        end
        @test err isa DecisionAnalysisError && err.problem === :deficiency
        @test occursin("deficiency", sprint(showerror, err))
    end

    @testset "certified no: mixed dominance and separating payoffs" begin
        rng = Xoshiro(20261124)
        @testset "model shapes" begin
            problem = DecisionProblem(random_payoff_table(rng, 4, 3; T=BigRat),
                                      random_credal_set(rng, 3, 5; T=BigRat))
            prog = mixed_dominance_model(problem, :a2)
            @test prog.name === :mixed_dominance
            @test JuMP.num_variables(prog.model) == 4 + 1
            @test length(prog.constraints.dominates) == 5
            @test_throws ArgumentError mixed_dominance_model(problem, :nope)
            e = Experiment(random_stochastic(rng, BigRat, 3, 4))
            f = Experiment(random_stochastic(rng, BigRat, 2, 4))
            sep = separating_payoff_model(f, e)
            @test sep.name === :separating_payoff
            @test JuMP.num_variables(sep.model) == 2 * 4 + 3
            @test size(sep.constraints.best) == (3, 2)
        end
        @testset "not E-admissible: a witness exactly when the answer is no" begin
            no = 0
            for _ in 1:40
                m, n, K = rand(rng, 2:5), rand(rng, 2:4), rand(rng, 1:4)
                problem = DecisionProblem(random_payoff_table(rng, m, n; T=BigRat),
                                          random_credal_set(rng, n, K; T=BigRat))
                admitted = decide(problem, EAdmissible()).chosen
                for a in feasible_actions(problem)
                    if a in admitted
                        @test_throws WitnessError sprint(io -> export_not_eadmissible_witness(io,
                                                                                              problem,
                                                                                              a))
                        continue
                    end
                    no += 1
                    for opt in (nothing, CDD, HIGHS)
                        function export_no()
                            return sprint(io -> export_not_eadmissible_witness(io, problem,
                                                                               a;
                                                                               optimizer=opt))
                        end
                        # Only the floating-point route may refuse (and is counted).
                        text = opt === HIGHS ? float_route(export_no) : export_no()
                        text === nothing && continue
                        w = check_not_eadmissible_witness(text)
                        @test w["action"] + 1 == findfirst(==(a), feasible_actions(problem))
                    end
                    # The optimum of the dual programme is minus that of the primal.
                    primal = solve_program(eadmissibility_model(problem, a))
                    dual = solve_program(mixed_dominance_model(problem, a))
                    @test dual.objective == -primal.objective > 0
                end
                # A "no" of the floating-point route is exact when certified.
                float = float_route(() -> decide(problem, EAdmissible(); optimizer=HIGHS))
                float === nothing && continue
                @test float.chosen == admitted
            end
            @test no > 20
            # The hedge that is maximal and still not E-admissible: the witness of the
            # Lean fixture, a half-half mixture of the two pure actions.
            weak = DecisionProblem(PayoffTable([:left, :right, :hedge], [:s1, :s2],
                                               [1//1 0; 0 1; 2//5 2//5]),
                                   CredalSet([:s1, :s2],
                                             [[1 // 1, 0 // 1], [0 // 1, 1 // 1]]))
            w = check_not_eadmissible_witness(sprint(io -> export_not_eadmissible_witness(io,
                                                                                          weak,
                                                                                          :hedge)))
            @test rational_vector(w["mixed_action"]) == [1 // 2, 1 // 2, 0]
            r = decide(weak, EAdmissible(); optimizer=HIGHS)
            @test r.chosen == [:left, :right] && r.details.exact[:hedge]
            @test_throws InexactCertificateError export_not_eadmissible_witness(IOBuffer(),
                                                                                DecisionProblem(PayoffTable([1.0 0.0;
                                                                                                             0.0 1.0]),
                                                                                                [0.5,
                                                                                                 0.5]),
                                                                                :a1)
            @test_throws ArgumentError export_not_eadmissible_witness(IOBuffer(), weak,
                                                                      :nope)
            @test_throws BeliefFormError export_not_eadmissible_witness(IOBuffer(),
                                                                        DecisionProblem(weak.payoff,
                                                                                        SampleBelief(2)),
                                                                        :hedge)
        end
        @testset "not a garbling: a witness exactly when the answer is no" begin
            no = 0
            for _ in 1:40
                S = rand(rng, 2:4)
                e = Experiment(random_stochastic(rng, BigRat, rand(rng, 2:3), S))
                f = rand(rng, Bool) ?
                    garble(e, random_stochastic(rng, BigRat, 2, size(e.likelihood, 1))) :
                    Experiment(random_stochastic(rng, BigRat, rand(rng, 2:3), S))
                if garbling_result(f, e).is_garbling
                    @test_throws WitnessError sprint(io -> export_not_garbling_witness(io,
                                                                                       f, e))
                    @test iszero(solve_program(separating_payoff_model(f, e)).objective)
                    continue
                end
                no += 1
                @test solve_program(separating_payoff_model(f, e)).objective > 0
                for opt in (nothing, CDD, HIGHS)
                    function export_no()
                        return sprint(io -> export_not_garbling_witness(io, f, e;
                                                                        optimizer=opt))
                    end
                    text = opt === HIGHS ? float_route(export_no) : export_no()
                    text === nothing && continue
                    w = check_not_garbling_witness(text)
                    @test rational_matrix(w["garbled"]) == f.likelihood
                    @test rational_matrix(w["experiment"]) == e.likelihood
                end
            end
            @test no > 10
            @test_throws InexactCertificateError export_not_garbling_witness(IOBuffer(),
                                                                             Experiment([0.8 0.2;
                                                                                         0.2 0.8]),
                                                                             Experiment([0.6 0.4;
                                                                                         0.4 0.6]))
            path = joinpath(mktempdir(), "no.json")
            e = Experiment([4//5 1//5; 1//5 4//5])
            @test export_not_garbling_witness(path, e, garble(e, [3//4 1//4; 1//4 3//4])) ==
                  path
            check_not_garbling_witness(read(path, String))
        end
    end

    @testset "witnesses by every route accepted by the Lean checker" begin
        proofs = joinpath(dirname(@__DIR__), "proofs")
        built = isdir(joinpath(proofs, ".lake", "build"))
        if lean_available() && Sys.which("lake") !== nothing && built
            fx = first(load_examples())
            flood = fixture_problem(fx, identity)
            credal = CredalSet(flood.payoff.states,
                               [rational_vector(q) for q in fx["credal"]])
            robust = DecisionProblem(flood.payoff, credal)
            gauge = fixture_experiment(fx["experiments"]["gauge"], identity)
            garbled = fixture_experiment(fx["experiments"]["garbled_gauge"], identity)
            mktempdir() do dir
                for (name, opt) in ("cdd" => CDD, "highs" => HIGHS)
                    files = [export_garbling_witness(joinpath(dir, "garbling_$name.json"),
                                                     garbled, gauge; optimizer=opt),
                             export_eadmissible_witness(joinpath(dir,
                                                                 "eadmissible_$name.json"),
                                                        robust, :raise_levee;
                                                        optimizer=opt),
                             export_not_eadmissible_witness(joinpath(dir,
                                                                     "not_eadmissible_$name.json"),
                                                            robust, :relocate_assets;
                                                            optimizer=opt),
                             export_not_garbling_witness(joinpath(dir,
                                                                  "not_garbling_$name.json"),
                                                         gauge, garbled; optimizer=opt)]
                    for file in files
                        code = run(Cmd(ignorestatus(`lake exe check_certificate $file`);
                                       dir=proofs)).exitcode
                        @test code == 0
                    end
                end
            end
        end
    end

    @testset "floating-point route: verified or refused, never unverified" begin
        @info "floating-point route (HiGHS with rounding) over this file" verified = FLOAT_ROUTE[:verified] refused = FLOAT_ROUTE[:refused]
        @test FLOAT_ROUTE[:verified] > 150
        # Refusals are allowed but must stay rare on these small problems.
        @test FLOAT_ROUTE[:refused] <= FLOAT_ROUTE[:verified] ÷ 10
    end
end
