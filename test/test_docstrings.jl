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

# A closure that assigns a variable it captures makes Julia box that variable. At best
# the function is then dynamically typed. At worst an inner function assigns a name that
# the enclosing function also uses and silently overwrites it: that gave wrong values in
# `adaptive_management` once and in `evsi(::MomentMatching)` once. No method of the
# package may contain a boxed variable.
@testset "no closure writes to a captured variable" begin
    boxed = String[]
    for name in names(DecisionAnalysis; all=true)
        isdefined(DecisionAnalysis, name) || continue
        f = getfield(DecisionAnalysis, name)
        f isa Function || continue
        for method in methods(f)
            method.module === DecisionAnalysis || continue
            code = try
                Base.uncompressed_ir(method)
            catch
                continue
            end
            for statement in code.code
                statement isa Expr && statement.head === :(=) || continue
                target = statement.args[1]
                target isa Core.SlotNumber || continue
                occursin("Core.Box", string(statement.args[2])) || continue
                push!(boxed,
                      string(method.name, " (", basename(string(method.file)), ":",
                             method.line, "): ", code.slotnames[target.id]))
            end
        end
    end
    @test isempty(unique(boxed))
    isempty(boxed) || foreach(println, sort(unique(boxed)))
end
