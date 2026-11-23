# The exception hierarchy of ADR 0013: every exception type this package defines, exported
# or not, subtypes the root `DecisionAnalysisError`, and the types keep their module and
# their unqualified names.
function owned_exception_types(M)
    types = Type[]
    for n in names(M; all=true)
        isdefined(M, n) || continue
        T = getglobal(M, n)
        T isa Type && T <: Exception && parentmodule(T) === M && push!(types, T)
    end
    return types
end

@testset "errors" begin
    owned = owned_exception_types(DecisionAnalysis)
    concrete = [InvalidBeliefError, InvalidPayoffError, InvalidExperimentError,
                LabelMismatchError, BeliefFormError, NoFeasibleActionError,
                InvalidBaselineError, MissingBaselineError, InexactCertificateError,
                WitnessError, LinearProgramError, ExactVerificationError,
                InvalidOutcomeError, MissingExtensionError]

    @testset "hierarchy" begin
        @test isabstracttype(DecisionAnalysisError)
        @test DecisionAnalysisError <: Exception
        # The search is not vacuous: it finds the root and exactly the concrete types.
        @test Set(owned) == Set(vcat(DecisionAnalysisError, concrete))
        for T in owned
            @test T <: DecisionAnalysisError
            @test string(T) == string(nameof(T))
        end
        for T in concrete
            @test isconcretetype(T)
            @test T in [getglobal(DecisionAnalysis, n) for n in names(DecisionAnalysis)]
        end
    end

    @testset "each type is raised and caught by the root" begin
        t = PayoffTable([:a, :b], [:s, :t], [1 0; 0 1])
        raised = Dict{Type,Function}(InvalidBeliefError => () -> FiniteBelief([0.5, 0.6]),
                                     InvalidPayoffError => () -> PayoffTable([1.0 Inf]),
                                     InvalidExperimentError => () -> Experiment([0.5 0.5]),
                                     LabelMismatchError => () -> DecisionProblem(t,
                                                                                 FiniteBelief([:s,
                                                                                               :u],
                                                                                              [0.5,
                                                                                               0.5])),
                                     BeliefFormError => () -> evpi(DecisionProblem(t,
                                                                                   ScenarioSet([:s]))),
                                     NoFeasibleActionError => () -> DecisionProblem(t,
                                                                                    [0.5,
                                                                                     0.5];
                                                                                    feasible=[false,
                                                                                              false]),
                                     InvalidBaselineError => () -> DecisionProblem(t,
                                                                                   [0.5,
                                                                                    0.5];
                                                                                   baseline=:c),
                                     MissingBaselineError => () -> gain_over_baseline(DecisionProblem(t,
                                                                                                      [0.5,
                                                                                                       0.5])),
                                     InexactCertificateError => () -> export_decision_certificate(IOBuffer(),
                                                                                                  DecisionProblem(t,
                                                                                                                  [0.5,
                                                                                                                   0.5])),
                                     WitnessError => () -> export_eadmissible_witness(IOBuffer(),
                                                                                      DecisionProblem(PayoffTable([1//1 0;
                                                                                                                   0 0]),
                                                                                                      [1 //
                                                                                                       1,
                                                                                                       0 //
                                                                                                       1]),
                                                                                      :a2),
                                     InvalidOutcomeError => () -> OutcomeTable([(;
                                                                                 draw=1,
                                                                                 action=:a,
                                                                                 replicate=1,
                                                                                 n=NaN)];
                                                                               outcomes=[:n]),
                                     MissingExtensionError => () -> throw(MissingExtensionError(:GAMRegression,
                                                                                                :GAM)),
                                     LinearProgramError => () -> throw(LinearProgramError(:deficiency,
                                                                                          :unbounded)),
                                     ExactVerificationError => () -> deficiency(Experiment([4//5 1//5;
                                                                                            1//5 4//5]),
                                                                                Experiment([2//3 1//7;
                                                                                            1//3 6//7]);
                                                                                optimizer=HiGHS.Optimizer,
                                                                                rounding_tol=0.2))
        @test Set(keys(raised)) == Set(concrete)
        for (T, f) in raised
            @test_throws T f()
            @test_throws DecisionAnalysisError f()
            err = try
                f()
            catch e
                e
            end
            message = sprint(showerror, err)
            @test startswith(message, string(nameof(T)))
            @test length(message) > length(string(nameof(T))) + 2
        end
    end

    @testset "messages carry the offending names" begin
        t = PayoffTable([:a, :b], [:s, :t], [1 0; 0 1])
        message(f) =
            try
                f()
                ""
            catch e
                sprint(showerror, e)
            end
        @test occursin("c", message(() -> DecisionProblem(t, [0.5, 0.5]; baseline=:c)))
        @test occursin("[:a, :b]",
                       message(() -> DecisionProblem(t, [0.5, 0.5]; baseline=:c)))
        @test occursin("gain_over_baseline",
                       message(() -> gain_over_baseline(DecisionProblem(t, [0.5, 0.5]))))
        @test occursin("ScenarioSet",
                       message(() -> evpi(DecisionProblem(t, ScenarioSet([:s])))))
        @test occursin("robust_evpi",
                       message(() -> evpi(DecisionProblem(t,
                                                          CredalSet([:s, :t],
                                                                    [[0.5, 0.5]])))))
        @test occursin("Float64",
                       message(() -> export_decision_certificate(IOBuffer(),
                                                                 DecisionProblem(t,
                                                                                 [0.5, 0.5]))))
    end

    @testset "invalid arguments stay ArgumentError" begin
        t = PayoffTable([:a, :b], [:s, :t], [1 0; 0 1])
        for f in (() -> Hurwicz(2), () -> CVaR(0),
                  () -> evppi(DecisionProblem(t, [0.5, 0.5]), [1]),
                  () -> decision_loss_bound(-1, 0.5),
                  () -> DecisionProblem(t, [0.5, 0.5]; feasible=[:zzz]))
            err = try
                f()
            catch e
                e
            end
            @test err isa ArgumentError
            @test !(err isa DecisionAnalysisError)
        end
    end

    @testset "errors from FiniteKernels pass through unchanged" begin
        X = FiniteKernels.FiniteSpace(:X, [:a, :b])
        @test_throws FiniteKernels.FiniteKernelsError FiniteBelief(FiniteKernels.state(X,
                                                                                       [0.5,
                                                                                        0.6]))
    end
end
