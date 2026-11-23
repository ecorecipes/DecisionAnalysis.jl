# Every exported name this package owns has a docstring (CLAUDE.md, "Style"). A name
# re-exported from another package is that package's to document, so it is skipped.
# `Base.which` names the module that owns a binding; `parentmodule` would throw on
# constants and `Union`s.
function owned_undocumented(M; allow=Symbol[])
    return filter(n -> Base.which(M, n) === M && n ∉ allow, Base.Docs.undocumented_names(M))
end

@testset "docstrings on every exported name" begin
    @test isempty(owned_undocumented(DecisionAnalysis))
    # The check is not vacuous, and the names the ecosystem reserves are not defined.
    @test length(names(DecisionAnalysis)) > 60
    for reserved in (:regret, :regrets, :compose, :otimes, :id, :mcopy, :delete, :braid,
                     :dom, :codom)
        @test reserved ∉ names(DecisionAnalysis)
        @test !isdefined(DecisionAnalysis, reserved)
    end
end
