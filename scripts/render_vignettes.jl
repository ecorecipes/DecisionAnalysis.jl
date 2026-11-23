# Render the vignettes to Markdown, HTML and PDF, running the code ONCE.
#
#   julia scripts/render_vignettes.jl                 # every vignette
#   julia scripts/render_vignettes.jl 04 07           # only those whose name starts so
#   julia scripts/render_vignettes.jl --no-execute    # redo HTML and PDF from the
#                                                     # Markdown already on disk
#
# The order, for each vignettes/NN_name/NN_name.qmd:
#
#   1. `quarto render NN_name.qmd --to gfm` executes the Julia code and writes
#      NN_name.md and the figures under NN_name_files/. This is the only step that
#      runs code.
#      The script then rewrites the `<img>` tags Quarto uses for figures as Markdown
#      images, which is a change of markup only.
#   2. pandoc converts NN_name.md to NN_name.html (self-contained: figures and style
#      embedded, mathematics as MathML).
#   3. pandoc converts NN_name.md to NN_name.pdf with lualatex.
#
# Steps 2 and 3 read the executed Markdown, so the three formats show the same
# numbers and the same figures by construction. pandoc is the one bundled with
# Quarto (`quarto pandoc`), so nothing but Quarto and a TeX distribution is needed.
#
# Fonts of the PDF: STIX Two Text and STIX Two Math for text and mathematics, and for
# code JuliaMono when the workspace has it at ../fonts/JuliaMono/ (as the sibling
# packages expect), otherwise the first installed of Menlo and DejaVu Sans Mono,
# otherwise the LaTeX default. The script prints which it used.

const ROOT = normpath(joinpath(@__DIR__, ".."))
const VIG = joinpath(ROOT, "vignettes")
const SUPPORT = joinpath(VIG, "_pandoc")
const NO_EXECUTE = "--no-execute" in ARGS
const WANTED = filter(a -> !startswith(a, "--"), ARGS)

function vignette_dirs()
    return sort(filter(readdir(VIG)) do d
                    return occursin(r"^\d+_", d) && isfile(joinpath(VIG, d, d * ".qmd")) &&
                           (isempty(WANTED) || any(w -> startswith(d, w), WANTED))
                end)
end

# TinyTeX installed by `quarto install tinytex` is not on PATH.
function tex_path()
    Sys.which("lualatex") === nothing || return ENV["PATH"]
    for base in (joinpath(homedir(), "Library", "TinyTeX", "bin"),
                 joinpath(homedir(), ".TinyTeX", "bin"),
                 joinpath(get(ENV, "APPDATA", homedir()), "TinyTeX", "bin"))
        isdir(base) || continue
        for d in readdir(base; join=true)
            any(startswith("lualatex"), readdir(d)) &&
                return string(d, Sys.iswindows() ? ';' : ':', ENV["PATH"])
        end
    end
    return error("lualatex not found: install a TeX distribution, for example with " *
                 "`quarto install tinytex`")
end

function font_installed(name::AbstractString)
    Sys.which("fc-list") === nothing && return false
    return !isempty(readchomp(`fc-list $name`))
end

# pandoc variables that choose the fonts of the PDF, and a description of the choice.
function pdf_fonts()
    text = font_installed("STIX Two Text") && font_installed("STIX Two Math")
    # STIX Two Text has no arrows or relation signs; they are taken from the maths font.
    vars = text ?
           ["-V", "mainfont=STIX Two Text", "-V", "sansfont=STIX Two Text", "-V",
            "mathfont=STIX Two Math", "-V", "mainfontfallback=STIX Two Math:", "-V",
            "sansfontfallback=STIX Two Math:"] : String[]
    used = text ? "STIX Two Text and STIX Two Math" : "the LaTeX default text fonts"
    juliamono = normpath(joinpath(ROOT, "..", "fonts", "JuliaMono"))
    if isfile(joinpath(juliamono, "JuliaMono-Regular.ttf"))
        append!(vars,
                ["-V", "monofont=JuliaMono-Regular", "-V",
                 "monofontoptions=Path=$(juliamono)/", "-V",
                 "monofontoptions=Extension=.ttf", "-V",
                 "monofontoptions=BoldFont=JuliaMono-Bold", "-V",
                 "monofontoptions=ItalicFont=JuliaMono-RegularItalic", "-V",
                 "monofontoptions=BoldItalicFont=JuliaMono-BoldItalic", "-V",
                 "monofontoptions=Scale=MatchLowercase"])
        return vars, used * "; JuliaMono from " * juliamono
    end
    for mono in ("Menlo", "DejaVu Sans Mono")
        font_installed(mono) || continue
        append!(vars,
                ["-V", "monofont=$mono", "-V", "monofontoptions=Scale=MatchLowercase"])
        return vars, used * "; $mono for code (JuliaMono not found at $juliamono)"
    end
    return vars,
           used * "; the LaTeX default monospace font (no JuliaMono, Menlo or " *
           "DejaVu Sans Mono found)"
end

"""
Split the Markdown that Quarto wrote into title, author and body. Quarto's gfm
output starts with `# Title`, the author on the next line, and a table of contents
as a list of links; pandoc is given the title and author as metadata and builds its
own table of contents, so all three are removed from the body.
"""
function split_markdown(text::AbstractString)
    lines = split(text, '\n')
    m = match(r"^#\s+(.*)$", lines[1])
    m === nothing && error("the Markdown does not start with a title heading")
    title = String(strip(m.captures[1]))
    i = 2
    author = ""
    if i <= length(lines) && !isempty(strip(lines[i]))
        author = String(strip(lines[i]))
        i += 1
    end
    while i <= length(lines) && isempty(strip(lines[i]))
        i += 1
    end
    # The table of contents: list items that are links to anchors, with wrapped lines.
    if i <= length(lines) && startswith(lines[i], "- [")
        while i <= length(lines) && !isempty(strip(lines[i]))
            i += 1
        end
    end
    return title, author, join(lines[i:end], '\n')
end

# Links between vignettes point at the Markdown; in the HTML and the PDF they point
# at the neighbouring file of the same format.
function retarget_links(body::AbstractString, ext::AbstractString)
    return replace(body,
                   r"\]\(\.\./(\d+_[A-Za-z0-9_]+)/\1\.(?:md|qmd)\)" => SubstitutionString("](../\\1/\\1.$ext)"))
end

function markdown_images(text::AbstractString)
    return replace(text, r"<img\s+src=\"([^\"]+)\"[^>]*/>"s => s"![](\1)")
end

function pandoc(dir, body, args)
    cmd = Cmd(`quarto pandoc --from gfm+tex_math_dollars $args`; dir)
    open(pipeline(cmd; stdout=stdout, stderr=stderr), "w") do io
        return write(io, body)
    end
    return nothing
end

function render(d::AbstractString, fontvars)
    dir = joinpath(VIG, d)
    times = Dict{String,Float64}()
    if !NO_EXECUTE
        times["md"] = @elapsed run(Cmd(`quarto render $(d * ".qmd") --to gfm --quiet`; dir))
        # A cold package cache prints its precompilation into the first cell. The
        # second run finds the cache warm, so this happens at most once per machine.
        if occursin("Precompiling packages", read(joinpath(dir, d * ".md"), String))
            @info "$d: precompilation output was captured; rendering once more"
            times["md"] += @elapsed run(Cmd(`quarto render $(d * ".qmd") --to gfm --quiet`;
                                            dir))
        end
    end
    if !NO_EXECUTE
        # Quarto writes a PNG figure as an HTML `<img>` tag with its pixel size. A
        # Markdown image is read by GitHub, Documenter and pandoc (for the PDF) alike.
        md = joinpath(dir, d * ".md")
        write(md, markdown_images(read(md, String)))
    end
    title, author, body = split_markdown(read(joinpath(dir, d * ".md"), String))
    meta = ["--metadata", "title=$title", "--metadata", "author=$author", "--standalone",
            "--toc", "--toc-depth=3",
            "--shift-heading-level-by=-1"]
    times["html"] = @elapsed pandoc(dir, retarget_links(body, "html"),
                                    [meta..., "--metadata", "lang=en-GB", "--to", "html5",
                                     "--embed-resources",
                                     "--mathml", "--css",
                                     joinpath(SUPPORT, "vignette.css"), "--output",
                                     d * ".html"])
    times["pdf"] = @elapsed pandoc(dir, retarget_links(body, "pdf"),
                                   [meta..., "--pdf-engine=lualatex", "--number-sections",
                                    "-V", "documentclass=scrartcl", "-V", "papersize=a4",
                                    "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V",
                                    "urlcolor=blue", fontvars..., "--include-in-header",
                                    joinpath(SUPPORT, "header.tex"), "--output",
                                    d * ".pdf"])
    return times
end

function main()
    ENV["PATH"] = tex_path()
    fontvars, fonts = pdf_fonts()
    println("PDF fonts: ", fonts)
    NO_EXECUTE ||
        run(`julia --project=$VIG -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'`)
    total = @elapsed for d in vignette_dirs()
        t = render(d, fontvars)
        println(rpad(d, 40),
                join(("$k $(round(t[k]; digits=1)) s"
                      for k in ("md", "html", "pdf")
                      if haskey(t, k)), ", "))
        for ext in (".md", ".html", ".pdf")
            isfile(joinpath(VIG, d, d * ext)) || error("$d$ext was not written")
        end
    end
    println("rendered ", length(vignette_dirs()), " vignette(s) in ",
            round(total; digits=1), " s")
    return nothing
end

main()
