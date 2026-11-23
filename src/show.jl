# Printing.

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
    v.lower === nothing || print(io, "; envelope [", v.lower, ", ", v.upper, "]")
    return print(io, ")")
end

# Numbers in a report: rationals as decimals when short, otherwise as they are.
_cell(x::Bool) = x ? "yes" : "-"
_cell(x::AbstractFloat) = string(round(x; sigdigits=6))
_cell(x::Integer) = string(x)
function _cell(x::Rational)
    isone(denominator(x)) && return string(numerator(x))
    return string(round(Float64(x); sigdigits=6))
end
_cell(x) = string(x)

function _print_grid(io::IO, header::Vector{String}, rows::Vector{Vector{String}})
    widths = [maximum(length, vcat(header[j], [r[j] for r in rows]))
              for j in eachindex(header)]
    function line(cells)
        return join((j == 1 ? rpad(c, widths[j]) : lpad(c, widths[j])
                     for (j, c) in enumerate(cells)), "  ")
    end
    println(io, "  ", line(header))
    println(io, "  ", join(("-"^w for w in widths), "  "))
    for r in rows
        println(io, "  ", line(r))
    end
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", r::DecisionReport)
    p = r.problem
    actions = feasible_actions(p)
    println(io, "DecisionReport: ", length(actions), " actions × ", size(p.payoff, 2),
            " states, ", nameof(typeof(p.belief)))
    println(io, "Baseline (no intervention): ",
            p.baseline === nothing ? "none set" : String(p.baseline))
    println(io)
    println(io, "Scores by rule (* marks the chosen actions)")
    header = vcat("action", [String(_rule_name(res.rule)) for res in r.results])
    rows = [vcat(String(a),
                 [_cell(res.scores[i]) * (a in res.chosen ? " *" : "  ")
                  for res in r.results]) for (i, a) in enumerate(actions)]
    _print_grid(io, header, rows)
    println(io)
    println(io, "First choice by rule")
    for res in r.results
        println(io, "  ", rpad(String(_rule_name(res.rule)), 20), first_choice(res),
                p.baseline === nothing ? "" :
                first_choice(res) == p.baseline ? "  [the baseline]" :
                "  [not the baseline]",
                length(res.chosen) > 1 ? "  (tied: " * join(res.chosen, ", ") * ")" : "")
    end
    if r.expected_gain !== nothing
        println(io)
        println(io, "Expected gain over the baseline (", p.baseline, ")")
        _print_grid(io, ["action", "expected gain"],
                    [[String(a), _cell(r.expected_gain[i])]
                     for (i, a) in enumerate(actions)])
    end
    println(io)
    println(io, "Regret profile")
    header = ["action", "max regret"]
    r.expected_regret === nothing || push!(header, "expected regret")
    rows = [vcat(String(a), _cell(r.max_regret[i]),
                 r.expected_regret === nothing ? String[] :
                 [_cell(r.expected_regret[i])]) for (i, a) in enumerate(actions)]
    _print_grid(io, header, rows)
    if !isempty(r.robust)
        println(io)
        println(io, "Robust sets over the credal set")
        for res in r.robust
            println(io, "  ", rpad(String(_rule_name(res.rule)), 20),
                    join(res.chosen, ", "))
        end
    end
    println(io)
    println(io, "Value of information")
    r.evpi === nothing || println(io, "  EVPI                ", _cell(r.evpi))
    r.evsi === nothing || println(io, "  EVSI                ", _cell(r.evsi))
    if r.robust_evpi !== nothing
        v = r.robust_evpi
        println(io, "  EVPI, Γ-maximin     ", _cell(v.value))
        println(io, "  EVPI envelope       [", _cell(v.lower), ", ", _cell(v.upper), "]")
    end
    r.evpi === nothing && r.robust_evpi === nothing &&
        println(io, "  not defined without probabilities")
    println(io)
    println(io, "Assumptions")
    for (i, a) in enumerate(r.assumptions)
        print(io, "  ", i, ". ", a)
        i == length(r.assumptions) || println(io)
    end
    return nothing
end

function Base.show(io::IO, r::DecisionReport)
    return print(io, "DecisionReport(", length(r.results), " rules, ",
                 length(feasible_actions(r.problem)), " actions)")
end
