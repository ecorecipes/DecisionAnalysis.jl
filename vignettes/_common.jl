# Shared presentation helpers for the vignettes: plot defaults, and a small Markdown
# table for summaries assembled by hand. Payoff tables, regret tables, rule results and
# reports need no helper: DecisionAnalysis displays them as tables itself.
# Nothing here computes a result; every number comes from DecisionAnalysis.

using Markdown: Markdown
using Plots: Plots

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")   # GR without a display
Plots.default(; size=(640, 380), dpi=200, legend=:outertopright, framestyle=:box,
              titlefontsize=11, guidefontsize=10, tickfontsize=9, legendfontsize=9,
              left_margin=4Plots.mm, bottom_margin=4Plots.mm, linewidth=2)

"""
    fmt(x; digits=2)

A number as the text of a table cell. Integers and whole rationals are printed as
integers, other rationals and floats as decimals rounded to `digits` places, `Bool`s
as yes or a dash, symbols as code, and vectors of symbols as a comma-separated list.
"""
fmt(x::Bool; digits=2) = x ? "yes" : "-"
fmt(x::Integer; digits=2) = string(x)
function fmt(x::Rational; digits=2)
    return isone(denominator(x)) ? string(numerator(x)) : fmt(Float64(x); digits)
end
function fmt(x::AbstractFloat; digits=2)
    r = round(x; digits)
    return r == round(r) ? string(Int(round(r))) : string(r)
end
fmt(x::Symbol; digits=2) = "`" * String(x) * "`"
fmt(x::AbstractVector{Symbol}; digits=2) = isempty(x) ? "none" : join(fmt.(x), ", ")
fmt(::Nothing; digits=2) = "none"
fmt(x; digits=2) = string(x)

"""
    exact(x)

A rational written out in full, for example `311//10` as `311/10`.
"""
exact(x::Integer) = string(x)
function exact(x::Rational)
    return isone(denominator(x)) ? string(numerator(x)) :
           string(numerator(x), "/", denominator(x))
end

"""
    md_table(header, rows; digits=2)

A Markdown table from a header (a vector) and rows (a vector of vectors). The first
column is left-aligned and the others right-aligned. Cells go through [`fmt`](@ref).
"""
function md_table(header, rows; digits=2)
    cell(x) = x isa AbstractString ? x : fmt(x; digits)
    io = IOBuffer()
    println(io, "| ", join(cell.(header), " | "), " |")
    println(io, "|:--", repeat("|--:", length(header) - 1), "|")
    for r in rows
        println(io, "| ", join(cell.(r), " | "), " |")
    end
    return Markdown.parse(String(take!(io)))
end

"""
    labelled_table(corner, row_labels, column_labels, M; digits=2)

A Markdown table of the matrix `M` with labelled rows and columns.
"""
function labelled_table(corner, row_labels, column_labels, M::AbstractMatrix; digits=2)
    return md_table(vcat(corner, collect(column_labels)),
                    [vcat(row_labels[i], collect(M[i, :])) for i in axes(M, 1)]; digits)
end

"""
    bullet(items)

A Markdown bullet list from a vector of strings.
"""
bullet(items) = Markdown.parse(join(("- " * s for s in items), "\n"))

"""
    grouped_bars(group_labels, series_labels, M; kwargs...)

A grouped bar chart: one cluster of bars per row of `M` (labelled by `group_labels`)
and one bar per column (labelled by `series_labels`).
"""
function grouped_bars(group_labels, series_labels, M::AbstractMatrix; kwargs...)
    n, k = size(M)
    width = 0.8 / k
    lo, hi = min(0, minimum(M)), max(0, maximum(M))
    pad = 0.08 * (hi - lo)
    plt = Plots.plot(; xticks=(1:n, String.(group_labels)),
                     ylims=(lo == 0 ? 0 : lo - pad, hi + pad), kwargs...)
    for j in 1:k
        Plots.bar!(plt, (1:n) .+ (j - (k + 1) / 2) * width, Float64.(M[:, j]);
                   bar_width=width, label=String(series_labels[j]), linewidth=0.5)
    end
    Plots.hline!(plt, [0]; color=:black, linewidth=1, label="")
    return plt
end

"""
    png_figure(plot)

Wrap a plot so that it is written as a PNG image and nothing else. Quarto would
otherwise store an SVG, which the PDF step can only read with an extra converter;
a PNG goes into the Markdown, the HTML and the PDF unchanged.
"""
struct PNGFigure{P}
    plot::P
end
png_figure(plot) = PNGFigure(plot)
Base.show(io::IO, m::MIME"image/png", f::PNGFigure) = show(io, m, f.plot)
Base.show(io::IO, ::MIME"text/plain", ::PNGFigure) = print(io, "figure")

"""
    voi_table(named_estimates; digits=3, truth=nothing)

A Markdown table of sample-based estimates (`name => VOIEstimate` pairs): the value,
its standard error and the two selection biases, each in its own column, so that no
error is folded into a value. `truth`, when given, adds a column of reference values.
"""
function voi_table(named; digits=3, truth=nothing)
    header = ["estimate", "value", "standard error", "act-now bias", "cell-noise bias"]
    truth === nothing || push!(header, "reference")
    rows = map(enumerate(named)) do (k, (name, e))
        row = Any[name, Float64(e.value), e.standard_error, e.bias.act_now,
                  e.bias.cells === nothing ? "-" : e.bias.cells]
        truth === nothing || push!(row, truth[k])
        return row
    end
    return md_table(header, rows; digits)
end
