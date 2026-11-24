using Aqua: Aqua

@testset "Aqua" begin
    # FiniteKernels is found through a `[sources]` URL, and Pkg does not follow the sources of a
    # URL dependency, so its unregistered dependencies are listed in `[deps]` although the code
    # does not load them.
    Aqua.test_all(DecisionAnalysis;
                  stale_deps=(; ignore=[:ResilienceCore, :StructuredPopulationCore]))
end
