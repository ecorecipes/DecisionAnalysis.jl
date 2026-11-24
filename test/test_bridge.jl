# The influence-diagram bridge lives in a weak-dependency extension. InfluenceDiagrams.jl
# brings GATlab and ACSets, which must stay out of this package's own environments, so
# its tests have an environment of their own (test/influence_diagrams) and run only
# with DECISIONANALYSIS_ID=1.

@testset "influence-diagram bridge" begin
    @test Base.get_extension(DecisionAnalysis, :DecisionAnalysisInfluenceDiagramsExt) ===
          nothing
    err = try
        normal_form(:not_a_diagram)
    catch e
        e
    end
    @test err isa MissingExtensionError
    @test occursin("using InfluenceDiagrams", sprint(showerror, err))
    # No environment of the package itself may hold Catlab, GATlab or ACSets.
    for file in ("Project.toml", joinpath("docs", "Project.toml"))
        text = read(joinpath(@__DIR__, "..", file), String)
        @test !occursin(r"^(Catlab|GATlab|ACSets)\s*="m, text)
    end
    if get(ENV, "DECISIONANALYSIS_ID", "") == "1"
        project = joinpath(@__DIR__, "influence_diagrams")
        script = joinpath(project, "runtests.jl")
        julia = Base.julia_cmd()
        setup = "using Pkg; Pkg.instantiate()"
        env = filter(p -> !startswith(first(p), "JULIA_LOAD_PATH") &&
                         first(p) != "JULIA_PROJECT", collect(ENV))
        run_in(cmd) = success(pipeline(setenv(cmd, env); stdout, stderr))
        @test run_in(`$julia --project=$project -e $setup`)
        @test run_in(`$julia --project=$project $script`)
    end
end
