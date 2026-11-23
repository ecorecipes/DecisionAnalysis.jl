using Documenter
using DecisionAnalysis

DocMeta.setdocmeta!(DecisionAnalysis, :DocTestSetup, :(using DecisionAnalysis);
                    recursive=true)

# Tutorials are rendered quarto vignettes copied into docs/src/tutorials by
# scripts/sync_vignettes.jl. The page list is built from the files on disk so
# it never has to be maintained by hand.
function tutorial_pages()
    dir = joinpath(@__DIR__, "src", "tutorials")
    isdir(dir) || return Pair{String,String}[]
    files = sort(filter(f -> endswith(f, ".md"), readdir(dir)))
    map(files) do f
        title = f
        for line in eachline(joinpath(dir, f))
            m = match(r"^#\s+(.*)", line)
            if m !== nothing
                title = String(strip(m.captures[1]))
                break
            end
        end
        return title => "tutorials/" * f
    end
end

# The API reference is split over several pages: one page of every docstring was close
# to Documenter's size limit for a page of HTML.
pages = Any["Home" => "index.md", "Certificates" => "certificates.md",
            "API Reference" => ["Beliefs, payoff tables and problems" => "api/problems.md",
                                "Aggregators, regret and rules" => "api/rules.md",
                                "Experiments and exact value of information" => "api/information.md",
                                "Sample-based analysis" => "api/samples.md",
                                "Linear programmes, reports, certificates and exceptions" => "api/programmes.md"]]
tutorials = tutorial_pages()
isempty(tutorials) || push!(pages, "Tutorials" => tutorials)

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
