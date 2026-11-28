# Tests for the SciMLBase weak-dependency extension (`ext/DecisionAnalysisSciMLBaseExt.jl`).
# Verified against closed-form logistic ODE solutions and a custom stochastic SciML problem.
using Test
using Random
using DecisionAnalysis
using SciMLBase
using OrdinaryDiffEqTsit5: Tsit5

@testset "DecisionAnalysisSciMLBaseExt" begin
    @test Base.get_extension(DecisionAnalysis, :DecisionAnalysisSciMLBaseExt) !== nothing
    @test !any(id -> id.name == "Catlab", keys(Base.loaded_modules))

    # Harvested logistic ODE: du/dt = (r - h) * u - (r / K) * u^2, with closed-form solution.
    function logistic_rhs(u, p, t)
        return [(p.r - p.h) * u[1] - (p.r / p.K) * u[1]^2]
    end
    function exact_logistic(u0, r, K, h, T)
        a = r - h
        b = r / K
        if abs(a) < 1e-12
            return u0 / (1 + b * u0 * T)
        end
        return a * u0 * exp(a * T) / (a + b * u0 * (exp(a * T) - 1))
    end

    p0 = (r=0.4, K=100.0, h=0.0)
    prob = ODEProblem(logistic_rhs, [25.0], (0.0, 6.0), p0; alg=Tsit5(), abstol=1e-9,
                      reltol=1e-9)

    @testset "the four adapter verbs" begin
        p1 = with_parameters(prob, (; r=0.5, K=120.0))
        @test p1.p == (r=0.5, K=120.0, h=0.0)
        @test prob.p == p0

        # Action as NamedTuple overriding parameters, as nothing/(;), and as a function.
        a_none = apply_action(p1, nothing)
        @test a_none.p == p1.p
        @test apply_action(p1, (;)).p == p1.p
        a_harv = apply_action(p1, (; p=(; h=0.15)))
        @test a_harv.p == (r=0.5, K=120.0, h=0.15)
        a_fn = apply_action(p1, pr -> SciMLBase.remake(pr; u0=[40.0]))
        @test a_fn.u0 == [40.0]

        sol = simulate(a_harv; rng=Xoshiro(1))
        o = outcome(sol)
        @test keys(o) == (:u1, :total)
        @test o.u1 == o.total
        @test isapprox(o.u1, exact_logistic(25.0, 0.5, 120.0, 0.15, 6.0); rtol=1e-6)

        # Validation errors on unmatched parameters or unknown action keys.
        @test_throws ArgumentError with_parameters(prob, (; nonexistent=1.0))
        @test_throws ArgumentError apply_action(prob, (; bad_key=1))
        @test_throws ArgumentError apply_action(prob, :not_an_action)
    end

    @testset "scalar and Dict-parameterised problems" begin
        dict_prob = ODEProblem((u, p, t) -> (p[:r] - p[:h]) * u, 10.0, (0.0, 2.0),
                               Dict(:r => 0.3, :h => 0.0); alg=Tsit5(), abstol=1e-9,
                               reltol=1e-9)
        q = apply_action(with_parameters(dict_prob, (; r=0.5)), (; p=(; h=0.1)))
        o = outcome(simulate(q; rng=Xoshiro(2)))
        @test keys(o) == (:final,)
        @test isapprox(o.final, 10.0 * exp(0.8); rtol=1e-6)
        # Original Dict is not mutated.
        @test dict_prob.p[:r] == 0.3 && dict_prob.p[:h] == 0.0
    end

    @testset "full outcomes -> DecisionProblem -> VOI pipeline" begin
        draws = (; r=[0.2, 0.35, 0.5, 0.65], K=[80.0, 100.0, 120.0, 90.0])
        actions = [:none => (;), :light => (; p=(; h=0.05)), :heavy => (; p=(; h=0.25))]
        tab = outcomes(prob, actions, draws; replicates=1, seed=7)
        @test run_counts(tab) == (runs=12, ok=12, failed=0, absent=0)
        @test action_labels(tab) == [:none, :light, :heavy]
        @test outcome_names(tab) == [:u1, :total]
        for (ai, h) in enumerate((0.0, 0.05, 0.25)), di in 1:4
            expected = exact_logistic(25.0, draws.r[di], draws.K[di], h, 6.0)
            @test isapprox(tab.values[ai, di, 1, 1], expected; rtol=1e-6)
        end
        dp = DecisionProblem(tab, IndividualsUtility(:total); baseline=:none)
        res = decide(dp, MaxExpectedUtility())
        @test first_choice(res) === :none
        @test Float64(evpi(dp)) >= 0.0
    end
end
