using Test
using DecisionAnalysis
using FiniteKernels: FiniteKernels
using JSON: JSON
using Random
using Tables: Tables

include("helpers.jl")

@testset "DecisionAnalysis" begin
    include("test_lp.jl")
    include("test_beliefs.jl")
    include("test_problem.jl")
    include("test_aggregators.jl")
    include("test_rules.jl")
    include("test_credal.jl")
    include("test_experiments.jl")
    include("test_voi.jl")
    include("test_fixtures.jl")
    include("test_sample.jl")
    include("test_outcomes.jl")
    include("test_utilities.jl")
    include("test_voi_sample.jl")
    include("test_gaussian_process.jl")
    include("test_standard_errors.jl")
    include("test_validation.jl")
    include("test_validation_improb.jl")
    include("test_sensitivity.jl")
    include("test_report.jl")
    include("test_design.jl")
    include("test_additions.jl")
    include("test_replications.jl")
    include("test_bridge.jl")
    include("test_certificates.jl")
    include("test_programs.jl")
    include("test_api.jl")
    include("test_lean_mirror.jl")
    include("test_errors.jl")
    include("test_docstrings.jl")
    include("test_aqua.jl")
end
