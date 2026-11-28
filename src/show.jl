# Printing. Compact one-line forms for every type, and tabular displays (plain text
# that fits 80 columns, Markdown and HTML) for payoff tables, regret tables, rule
# results and the decision report.

function Base.show(io::IO, b::FiniteBelief)
    return print(io, "FiniteBelief(",
                 join(("$s => $p"
                       for (s, p) in zip(b.states, b.probabilities)),
                      ", "), ")")
end

function Base.show(io::IO, b::CredalSet)
    return print(io, "CredalSet(", length(b.points), " point",
                 length(b.points) == 1 ? "" : "s", " over ", length(b.states), " states)")
end

Base.show(io::IO, b::ScenarioSet) = print(io, "ScenarioSet(", b.states, ")")

function Base.show(io::IO, b::SampleBelief)
    return print(io, "SampleBelief(", length(b), " draws, parameters ",
                 isempty(b.draws) ? "none" : join(keys(b.draws), ", "),
                 ", effective sample size ",
                 round(Float64(effective_sample_size(b)); sigdigits=4), ")")
end

function Base.show(io::IO, t::PayoffTable)
    return print(io, "PayoffTable{", eltype(t), "}(", size(t, 1), " actions × ",
                 size(t, 2), " states)")
end

function Base.show(io::IO, p::DecisionProblem)
    return print(io, "DecisionProblem(", count(p.feasible), " feasible of ",
                 size(p.payoff, 1), " actions × ", size(p.payoff, 2), " states, ",
                 nameof(typeof(p.belief)), ")")
end

function Base.show(io::IO, e::Experiment)
    return print(io, "Experiment{", eltype(e), "}(", size(e.likelihood, 1), " signals × ",
                 size(e.likelihood, 2), " states, cost ", e.cost, ")")
end

Base.show(io::IO, r::DecisionRule) = print(io, r.name)

function Base.show(io::IO, r::DecisionResult)
    return print(io, "DecisionResult(", _rule_name(r.rule), ": ",
                 isempty(r.chosen) ? "no action chosen" : join(r.chosen, ", "), ")")
end

function Base.show(io::IO, v::RobustVOI)
    print(io, "RobustVOI(", v.quantity, " under ", _rule_name(v.rule), ": ", v.value)
    v.lower === nothing ||
        print(io, "; envelope [", v.lower, ", ", v.upper, "], upper end at the prior ",
              v.upper_prior)
    return print(io, ")")
end

function Base.show(io::IO, p::DecisionProgram{T}) where {T}
    return print(io, "DecisionProgram{", T, "}(", p.name, ": ",
                 JuMP.num_variables(p.model), " variables, ",
                 JuMP.num_constraints(p.model; count_variable_in_set_constraints=false),
                 " constraints)")
end

function Base.show(io::IO, s::ProgramSolution)
    print(io, "ProgramSolution(", s.status)
    s.objective === nothing || print(io, ", objective ", s.objective)
    return print(io, ", ", s.route, s.exact ? ", exact)" : ", not exact)")
end

function Base.show(io::IO, r::GarblingResult)
    return print(io, "GarblingResult(", r.is_garbling ? "a garbling" : "not a garbling",
                 r.exact ? ", exact" : ", numerical", ", ", r.route,
                 r.separating === nothing ? "" : ", with a separating decision problem",
                 ")")
end

function Base.show(io::IO, r::DecisionReport)
    return print(io, "DecisionReport(", length(r.results), " rules, ",
                 length(feasible_actions(r.problem)), " actions)")
end

# Numbers in a display: rationals as decimals when short, otherwise as they are.
_cell(x::Bool) = x ? "yes" : "-"
_cell(x::AbstractFloat) = string(round(x; sigdigits=6))
_cell(x::Integer) = string(x)
function _cell(x::Rational)
    isone(denominator(x)) && return string(numerator(x))
    f = Float64(x)
    return isfinite(f) ? string(round(f; sigdigits=6)) : string(x)
end
_cell(x::Symbol) = String(x)
_cell(::Nothing) = "-"
_cell(x) = string(x)

# A display is a list of blocks, rendered as plain text, Markdown or HTML.
# ------------------------------------------------------------------------

# A table. The first column is left-aligned and the others right-aligned unless
# `left` marks them; `code` marks the columns whose cells are labels (actions, rules),
# and `code_header` says that the header cells after the first are labels too.
struct _Grid
    header::Vector{String}
    rows::Vector{Vector{String}}
    left::Vector{Bool}
    code::Vector{Bool}
    code_header::Bool
end
function _Grid(header, rows; left=Int[], code=[1], code_header::Bool=false)
    n = length(header)
    return _Grid(String.(header), [String.(r) for r in rows],
                 [j == 1 || j in left for j in 1:n], [j in code for j in 1:n], code_header)
end

struct _Title
    text::String
end
struct _Para
    text::String
end
struct _Items
    items::Vector{String}
end

const _WIDTH = 80

# Break `text` into lines of at most `width` characters at spaces.
function _wrap(text::AbstractString, width::Int)
    lines = String[]
    line = ""
    for word in split(text)
        if isempty(line)
            line = String(word)
        elseif length(line) + 1 + length(word) <= width
            line = line * " " * word
        else
            push!(lines, line)
            line = String(word)
        end
    end
    push!(lines, line)
    return lines
end

# Column blocks of a grid that each fit `width`: the first column is repeated in
# every block.
function _column_blocks(widths::Vector{Int}, width::Int)
    blocks = Vector{Int}[]
    current = Int[]
    used = 2 + widths[1]
    for j in 2:length(widths)
        if !isempty(current) && used + 2 + widths[j] > width
            push!(blocks, current)
            current = Int[]
            used = 2 + widths[1]
        end
        push!(current, j)
        used += 2 + widths[j]
    end
    (isempty(current) && !isempty(blocks)) || push!(blocks, current)
    return blocks
end

function _plain(io::IO, g::_Grid)
    widths = [maximum(length, vcat(g.header[j], [r[j] for r in g.rows]))
              for j in eachindex(g.header)]
    for (k, block) in enumerate(_column_blocks(widths, _WIDTH))
        k == 1 || println(io)
        cols = vcat(1, block)
        function line(cells)
            return rstrip(join((g.left[j] ? rpad(cells[j], widths[j]) :
                                lpad(cells[j], widths[j]) for j in cols), "  "))
        end
        println(io, "  ", line(g.header))
        println(io, "  ", join(("-"^widths[j] for j in cols), "  "))
        for r in g.rows
            println(io, "  ", line(r))
        end
    end
    return nothing
end
function _plain(io::IO, t::_Title)
    for line in split(t.text, '\n')
        foreach(l -> println(io, l), _wrap(line, _WIDTH))
    end
    return nothing
end
function _plain(io::IO, p::_Para)
    foreach(l -> println(io, "  ", l), _wrap(p.text, _WIDTH - 2))
    return nothing
end
function _plain(io::IO, list::_Items)
    pad = ndigits(length(list.items)) + 2
    for (i, item) in enumerate(list.items)
        lines = _wrap(item, _WIDTH - 2 - pad)
        println(io, "  ", rpad(string(i, "."), pad), lines[1])
        foreach(l -> println(io, "  ", " "^pad, l), lines[2:end])
    end
    return nothing
end

# Plain text: a blank line before every title but the first, no trailing newline.
function _render_plain(io::IO, blocks)
    buffer = IOBuffer()
    for (i, b) in enumerate(blocks)
        b isa _Title && i > 1 && println(buffer)
        _plain(buffer, b)
    end
    return print(io, rstrip(String(take!(buffer))))
end

_md_escape(s::AbstractString) = replace(s, "|" => "\\|", "*" => "\\*", "_" => "\\_")
_md_code(s::AbstractString) = isempty(s) ? "" : "`" * replace(s, "|" => "\\|") * "`"
function _md_cell(g::_Grid, j::Int, s::AbstractString; header::Bool=false)
    code = header ? (g.code_header && j > 1) : g.code[j]
    return code ? _md_code(s) : _md_escape(s)
end

function _markdown(io::IO, g::_Grid)
    cols = eachindex(g.header)
    println(io, "| ", join((_md_cell(g, j, g.header[j]; header=true) for j in cols), " | "),
            " |")
    println(io, "|", join((g.left[j] ? ":--" : "--:" for j in cols), "|"), "|")
    for r in g.rows
        println(io, "| ", join((_md_cell(g, j, r[j]) for j in cols), " | "), " |")
    end
    return println(io)
end
function _markdown(io::IO, t::_Title)
    return println(io,
                   join(("**" * _md_escape(l) * "**" for l in split(t.text, '\n')),
                        "  \n"), "\n")
end
_markdown(io::IO, p::_Para) = println(io, _md_escape(p.text), "\n")
function _markdown(io::IO, list::_Items)
    for (i, item) in enumerate(list.items)
        println(io, i, ". ", _md_escape(item))
    end
    return println(io)
end
_render_markdown(io::IO, blocks) = foreach(b -> _markdown(io, b), blocks)

_html_escape(s::AbstractString) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")
function _html_cell(g::_Grid, j::Int, s::AbstractString; header::Bool=false)
    code = header ? (g.code_header && j > 1) : g.code[j]
    text = _html_escape(s)
    return code && !isempty(s) ? "<code>" * text * "</code>" : text
end

function _html(io::IO, g::_Grid)
    cols = eachindex(g.header)
    align(j) = g.left[j] ? "left" : "right"
    println(io, "<table>")
    println(io, "<thead><tr>",
            join(("<th style=\"text-align: $(align(j));\">" *
                  _html_cell(g, j, g.header[j]; header=true) * "</th>" for j in cols)),
            "</tr></thead>")
    println(io, "<tbody>")
    for r in g.rows
        println(io, "<tr>",
                join(("<td style=\"text-align: $(align(j));\">" * _html_cell(g, j, r[j]) *
                      "</td>" for j in cols)), "</tr>")
    end
    return println(io, "</tbody>\n</table>")
end
function _html(io::IO, t::_Title)
    return println(io, "<p><strong>",
                   join((_html_escape(l) for l in split(t.text, '\n')), "<br>"),
                   "</strong></p>")
end
_html(io::IO, p::_Para) = println(io, "<p>", _html_escape(p.text), "</p>")
function _html(io::IO, list::_Items)
    println(io, "<ol>")
    foreach(item -> println(io, "<li>", _html_escape(item), "</li>"), list.items)
    return println(io, "</ol>")
end
_render_html(io::IO, blocks) = foreach(b -> _html(io, b), blocks)

# The blocks of each displayed type
# ---------------------------------

# A labelled matrix: one row per action, one column per state.
function _matrix_grid(corner::AbstractString, rows, columns, M::AbstractMatrix)
    return _Grid(vcat(corner, String.(columns)),
                 [vcat(String(rows[i]), [_cell(M[i, j]) for j in axes(M, 2)])
                  for i in axes(M, 1)]; code_header=true)
end

function _blocks(t::PayoffTable)
    blocks = Any[_Title("PayoffTable{$(eltype(t))}: $(size(t, 1)) actions × " *
                        "$(size(t, 2)) states"),
                 _matrix_grid("action", t.actions, t.states, t.Q)]
    if t.standard_errors !== nothing
        push!(blocks, _Title("Standard errors"),
              _matrix_grid("action", t.actions, t.states, t.standard_errors))
    end
    single_replicate(t) &&
        push!(blocks,
              _Para("Flag: some cells hold a single simulator run, not a conditional " *
                    "expectation; the best reported value and the value with " *
                    "information from this table are each biased upwards."))
    return blocks
end

function _blocks(t::RegretTable)
    return Any[_Title("RegretTable{$(eltype(t))}: " *
                      (t.relative ? "relative regret" : "regret") *
                      ", $(size(t, 1)) actions × $(size(t, 2)) states"),
               _matrix_grid("action", t.actions, t.states, t.R)]
end

_sense_words(r::DecisionResult) = r.sense === :max ? "higher is better" : "lower is better"

function _blocks(r::DecisionResult)
    actions = r.actions
    chosen = [_cell(a in r.chosen) for a in actions]
    name = String(_rule_name(r.rule))
    blocks = Any[]
    if r.rule isa ParetoFront
        objectives = String.(r.details.objectives)
        push!(blocks,
              _Title("DecisionResult: $name over $(length(objectives)) objectives, " *
                     "each summarised by $(nameof(typeof(r.rule.aggregator)))"),
              _Grid(vcat("action", objectives, "on the front"),
                    [vcat(String(a), [_cell(r.scores[i, j]) for j in axes(r.scores, 2)],
                          chosen[i]) for (i, a) in enumerate(actions)];
                    code=[1]))
    elseif r.scores === nothing
        header = ["action"]
        columns = Vector{String}[]
        if haskey(r.details, :lower)
            push!(header, "lower expectation", "upper expectation")
            push!(columns, _cell.(r.details.lower), _cell.(r.details.upper))
        end
        push!(header, "kept")
        push!(columns, chosen)
        push!(blocks, _Title("DecisionResult: $name (set-valued)"),
              _Grid(header,
                    [vcat(String(a), [c[i] for c in columns])
                     for (i, a) in enumerate(actions)]))
    else
        push!(blocks, _Title("DecisionResult: $name ($(_sense_words(r)))"),
              _Grid(["action", "score", "chosen"],
                    [[String(a), _cell(r.scores[i]), chosen[i]]
                     for (i, a) in enumerate(actions)]))
    end
    if !(r.rule isa ParetoFront) && !(r.scores === nothing)
        push!(blocks,
              _Para("First choice: " *
                    (isempty(r.chosen) ? "none (no action chosen)" :
                     String(first_choice(r))) *
                    (length(r.chosen) > 1 ? " (tied: " * join(r.chosen, ", ") * ")" : "")))
    end
    haskey(r.details, :note) && push!(blocks, _Para("Note: " * r.details.note))
    haskey(r.details, :sample_note) &&
        push!(blocks, _Para("Note: " * r.details.sample_note))
    if haskey(r.details, :exact) && !all(values(r.details.exact))
        unsure = [a for a in actions if !r.details.exact[a]]
        push!(blocks,
              _Para("Numerical, not exact, for: " * join(unsure, ", ") * " (route: " *
                    String(r.details.route) * ")."))
    end
    return blocks
end

function _baseline_mark(r::DecisionReport, res::DecisionResult)
    r.problem.baseline === nothing && return ""
    return first_choice(res) == r.problem.baseline ? "[the baseline]" :
           "[not the baseline]"
end

function _blocks(r::DecisionReport)
    p = r.problem
    actions = feasible_actions(p)
    labels = String.(actions)
    blocks = Any[_Title("DecisionReport: $(length(actions)) actions × " *
                        "$(size(p.payoff, 2)) states, $(nameof(typeof(p.belief)))\n" *
                        "Baseline (no intervention): " *
                        (p.baseline === nothing ? "none set" : String(p.baseline)))]

    push!(blocks, _Title("Scores by rule (* marks the chosen actions)"),
          _Grid(vcat("action", [String(_rule_name(res.rule)) for res in r.results]),
                [vcat(labels[i],
                      [_cell(res.scores[i]) * (a in res.chosen ? " *" : "  ")
                       for res in r.results]) for (i, a) in enumerate(actions)]))

    header = ["rule", "first choice"]
    p.baseline === nothing || push!(header, "baseline")
    any(res -> length(res.chosen) > 1, r.results) && push!(header, "tied")
    rows = map(r.results) do res
        row = [String(_rule_name(res.rule)), _cell(first_choice(res))]
        p.baseline === nothing || push!(row, _baseline_mark(r, res))
        "tied" in header &&
            push!(row, length(res.chosen) > 1 ? join(res.chosen, ", ") : "")
        return row
    end
    push!(blocks, _Title("First choice by rule"),
          _Grid(header, rows; left=collect(2:length(header)), code=[1, 2]),
          _Para("Minimax regret depends on the menu of actions: adding or removing " *
                "an action, even one that is never chosen, can change its choice."))

    mc = r.monte_carlo
    if mc !== nothing
        mc_header = ["action", "expected utility", "standard error", "simulation only"]
        mc.expected_gain === nothing ||
            push!(mc_header, "gain over baseline", "standard error of gain")
        mc_rows = map(eachindex(actions)) do i
            e = mc.expected_utility[i]
            row = [labels[i], _cell(e.value), _cell(e.standard_error),
                   _cell(e.simulation_error)]
            mc.expected_gain === nothing ||
                push!(row, _cell(mc.expected_gain[i].value),
                      _cell(mc.expected_gain[i].standard_error))
            return row
        end
        push!(blocks,
              _Title("Monte Carlo standard errors (shown beside the values, never " *
                     "folded into them)"), _Grid(mc_header, mc_rows),
              _Para("Standard error: the draws taken as a random sample of the " *
                    "belief. Simulation only: the simulator's noise with the draws " *
                    "held fixed (- when the table has no standard errors)."))
        mc.single_replicate &&
            push!(blocks,
                  _Para("Flag: some cells hold a single simulator run, not a " *
                        "conditional expectation; the act-now value and the value " *
                        "with information are each biased upwards."))
        if mc.variance !== nothing
            push!(blocks, _Title("Variance of the payoff across draws"),
                  _Grid(["action", "total", "simulation", "parameter", "parameter share"],
                        [[String(v.action), _cell(v.total), _cell(v.simulation),
                          _cell(v.parameter), _cell(v.parameter_share)]
                         for v in mc.variance]))
        end
    end

    if r.expected_gain !== nothing
        push!(blocks, _Title("Expected gain over the baseline ($(p.baseline))"),
              _Grid(["action", "expected gain"],
                    [[labels[i], _cell(r.expected_gain[i])] for i in eachindex(actions)]))
    elseif r.gain_range !== nothing
        what = p.belief isa CredalSet ?
               "Range of the expected gain over the baseline " *
               "($(p.baseline)), over the credal set" :
               "Range of the gain over the baseline ($(p.baseline)), over the scenarios"
        push!(blocks, _Title(what),
              _Grid(["action", "lowest gain", "highest gain"],
                    [[labels[i], _cell(r.gain_range.lower[i]),
                      _cell(r.gain_range.upper[i])] for i in eachindex(actions)]))
    end

    regret_header = ["action", "max regret"]
    r.expected_regret === nothing || push!(regret_header, "expected regret")
    push!(blocks, _Title("Regret profile"),
          _Grid(regret_header,
                [vcat(labels[i], _cell(r.max_regret[i]),
                      r.expected_regret === nothing ? String[] :
                      [_cell(r.expected_regret[i])]) for i in eachindex(actions)]))

    if !isempty(r.robust)
        push!(blocks, _Title("Robust sets over the credal set"),
              _Grid(["rule", "actions kept"],
                    [[String(_rule_name(res.rule)), join(res.chosen, ", ")]
                     for res in r.robust]; left=[2], code=[1, 2]))
    end

    push!(blocks, _Title("Value of information"))
    rows = Vector{String}[]
    r.evpi === nothing || push!(rows, ["EVPI", _cell(r.evpi), ""])
    if mc !== nothing
        estimates = Any["EVPI" => mc.evpi]
        mc.evpi_regression === nothing ||
            push!(estimates, "EVPI, all-parameter regression" => mc.evpi_regression)
        for q in mc.evppi
            push!(estimates, "EVPPI of $(q.name) ($(q.estimate.method))" => q.estimate)
        end
        push!(blocks,
              _Grid(["estimate", "value", "standard error", "act-now bias",
                     "cell-noise bias"],
                    [[name, _cell(e.value), _cell(e.standard_error),
                      _cell(e.bias.act_now), _cell(e.bias.cells)]
                     for (name, e) in estimates]; code=Int[]),
              _Para("Errors and biases are beside the values, not in them. The " *
                    "act-now bias is the upward bias of the best reported value " *
                    "from maximising sample means: the chosen action is on average " *
                    "worth less than its score. The cell-noise bias is the upward " *
                    "bias of the perfect-information value from maximising noisy " *
                    "cells. A value of information is a difference of two such " *
                    "terms, so the sign of its bias is not determined."))
        filter!(row -> row[1] != "EVPI", rows)
    end
    for e in r.experiments
        push!(rows,
              ["EVSI of $(e.name)", _cell(e.evsi),
               e.can_change_decision ? "can change the decision" :
               "this experiment cannot change the decision"])
    end
    for q in r.partitions
        push!(rows, ["EVPPI of $(q.name)", _cell(q.evppi), ""])
    end
    if r.robust_evpi !== nothing
        v = r.robust_evpi
        push!(rows, ["EVPI, Γ-maximin", _cell(v.value), ""],
              ["EVPI envelope", "[" * _cell(v.lower) * ", " * _cell(v.upper) * "]", ""])
    end
    if isempty(rows)
        mc === nothing && push!(blocks, _Para("not defined without probabilities"))
    else
        noted = any(row -> !isempty(row[3]), rows)
        push!(blocks,
              _Grid(noted ? ["quantity", "value", "note"] : ["quantity", "value"],
                    noted ? rows : [row[1:2] for row in rows]; left=[3], code=Int[]))
    end
    if any(e -> !iszero(e.cost), r.experiments)
        push!(blocks, _Title("Experiments net of their cost"),
              _Grid(["experiment", "EVSI", "cost", "EVSI − cost"],
                    [[String(e.name), _cell(e.evsi), _cell(e.cost), _cell(e.net)]
                     for e in r.experiments]))
    end

    for item in r.profiles
        prof = item.profile
        push!(blocks,
              _Title("Value of $(item.name) by outcome (acting now: " *
                     "$(prof.act_now_action))"),
              _Grid(["signal", "probability", "best action", "value of this outcome"],
                    [[String(o.signal), _cell(o.probability), join(o.best, ", "),
                      _cell(o.value)] for o in prof.outcomes]; left=[3], code=[1, 3]),
              _Para("EVSI " * _cell(prof.evsi) * "; standard deviation of the value " *
                    "over outcomes " * _cell(prof.standard_deviation) *
                    "; probability that the study changes nothing " *
                    _cell(prof.probability_zero) * "."))
    end
    if !isempty(r.estimation)
        push!(blocks, _Title("Value of information for estimation (variance reduction)"),
              _Grid(["source", "quantity", "prior variance", "expected reduction",
                     "share"],
                    [[String(e.source), String(e.quantity), _cell(e.prior_variance),
                      _cell(e.reduction), _cell(e.share)] for e in r.estimation];
                    code=[1, 2]))
    end

    if r.pareto !== nothing
        front = r.pareto
        objectives = String.(front.details.objectives)
        push!(blocks,
              _Title("Several objectives: Pareto front, each objective summarised by " *
                     string(nameof(typeof(front.rule.aggregator)))),
              _Grid(vcat("action", objectives, "on the front"),
                    [vcat(labels[i],
                          [_cell(front.scores[i, j]) for j in axes(front.scores, 2)],
                          _cell(a in front.chosen)) for (i, a) in enumerate(actions)]),
              _Para("Not dominated: " * join(front.chosen, ", ") * "."))
    end

    push!(blocks, _Title("Assumptions"), _Items(r.assumptions))
    return blocks
end

# One method per type: a `Union` signature would be ambiguous with Base's text/plain
# method for arrays, which a `RegretTable` is.
for T in (:PayoffTable, :RegretTable, :DecisionResult, :DecisionReport)
    @eval begin
        Base.show(io::IO, ::MIME"text/plain", x::$T) = _render_plain(io, _blocks(x))
        Base.show(io::IO, ::MIME"text/markdown", x::$T) = _render_markdown(io, _blocks(x))
        Base.show(io::IO, ::MIME"text/html", x::$T) = _render_html(io, _blocks(x))
    end
end
