using Documenter
using DecisionAnalysis

DocMeta.setdocmeta!(DecisionAnalysis, :DocTestSetup, :(using DecisionAnalysis);
                    recursive=true)

pages = Any["Home" => "index.md", "Certificates" => "certificates.md",
            "API Reference" => "api.md"]

# Source links need a commit to point at. A checkout without commits (a fresh
# scaffold, a tarball) builds without them instead of failing.
has_commit = success(pipeline(`git -C $(@__DIR__) rev-parse HEAD`; stderr=devnull,
                              stdout=devnull))
remote_kw = has_commit ? (; repo=Remotes.GitHub("ecorecipes", "DecisionAnalysis.jl")) :
            (; remotes=nothing)

# Strict: no `warnonly`, so a docstring left out of the manual, a broken `@ref` or a
# failing doctest fails the build.
makedocs(;
         remote_kw...,
         modules=[DecisionAnalysis],
         sitename="DecisionAnalysis.jl",
         authors="Simon Frost",
         format=Documenter.HTML(;
                                prettyurls=get(ENV, "CI", "false") == "true",
                                canonical="https://ecorecipes.github.io/DecisionAnalysis.jl",
                                repolink="https://github.com/ecorecipes/DecisionAnalysis.jl",
                                edit_link="main"),
         pages=pages)

"--no-deploy" in ARGS || deploydocs(;
                                    repo="github.com/ecorecipes/DecisionAnalysis.jl.git",
                                    devbranch="main")
