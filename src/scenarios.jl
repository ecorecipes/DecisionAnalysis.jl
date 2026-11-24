# Scenario design: sets of scenarios built from axes that are decisions or uncertainties,
# after Runge et al. (2024, "Scenario design for infectious disease projections:
# integrating concepts from decision analysis and experimental design", *Epidemics*).
# A design is the plan of a table of runs; what it can answer follows from the kinds of
# its axes, and the analyses are those of the rest of the package.

const _DESIGN_PURPOSES = (:decision_making, :value_of_information, :sensitivity_analysis,
                          :situational_awareness, :horizon_scanning, :forecasting)

"""
    ScenarioAxis(name, levels; kind, weights=nothing)

One axis of a scenario design: a factor that differs between scenarios. `kind` is
`:decision` for an intervention, under the control of the decision maker the design
is made for, or `:uncertainty` for a factor that nobody controls. The same factor can
be either, depending on the audience (vaccine uptake is a decision for an agency that
can campaign for it and an uncertainty for a hospital), so the kind is stated, never
guessed.

`levels` is a list of labels, or of `label => value` pairs (or a named tuple) when a
level stands for a value that a model takes (`:low => 2.2`). Without values, the
labels are the values. `weights` are probabilities of the levels of an uncertainty
axis; they are needed only by a forecasting design and by expectations over the
scenarios, and most designs have none. See [`decision_axis`](@ref),
[`uncertainty_axis`](@ref) and [`ScenarioDesign`](@ref).
"""
struct ScenarioAxis{W}
    name::Symbol
    kind::Symbol
    levels::Vector{Symbol}
    values::Vector{Any}
    weights::W
    function ScenarioAxis(name, levels; kind::Symbol, weights=nothing)
        kind in (:decision, :uncertainty) ||
            throw(ArgumentError("kind must be :decision or :uncertainty, got :$kind"))
        pairs_ = levels isa NamedTuple ? [k => v for (k, v) in pairs(levels)] :
                 [l isa Pair ? Symbol(first(l)) => last(l) : Symbol(string(l)) => l
                  for l in levels]
        isempty(pairs_) && throw(ArgumentError("the axis $name has no level"))
        labels = Symbol[first(p) for p in pairs_]
        allunique(labels) ||
            throw(ArgumentError("the levels of the axis $name must be distinct: $labels"))
        w = weights
        if weights !== nothing
            kind === :uncertainty ||
                throw(ArgumentError("only an uncertainty axis has weights; $name is a " *
                                    "decision"))
            length(weights) == length(labels) ||
                throw(ArgumentError("$(length(weights)) weights for the " *
                                    "$(length(labels)) levels of $name"))
            w = _normalise_weights(collect(weights))
        end
        return new{typeof(w)}(Symbol(name), kind, labels, Any[last(p) for p in pairs_], w)
    end
end

"""
    decision_axis(name, levels)

A [`ScenarioAxis`](@ref) of interventions. Include no intervention among the levels:
it is the counterfactual that the others are compared with.
"""
decision_axis(name, levels) = ScenarioAxis(name, levels; kind=:decision)

"""
    uncertainty_axis(name, levels; weights=nothing)

A [`ScenarioAxis`](@ref) of a factor outside anyone's control, with optional
probabilities of its levels.
"""
uncertainty_axis(name, levels; weights=nothing) = ScenarioAxis(name, levels;
                                                               kind=:uncertainty, weights)

"""
    ScenarioDesign(axes...; purpose=nothing, audience="")

A scenario design: the scenarios are the combinations of the levels of its axes (a
full factorial; see [`fractional_design`](@ref) for a part of it). What a design can
answer follows from the kinds of its axes ([`design_class`](@ref)):

| axes | class | purposes |
|:--|:--|:--|
| decisions only | `:decision` | `:decision_making`: compare interventions |
| a decision and an uncertainty | `:value_of_information` | `:value_of_information`: is the better intervention the same whatever the uncertainty? |
| uncertainties only | `:uncertainty` | `:sensitivity_analysis`, `:situational_awareness`, `:horizon_scanning`, `:forecasting` |

The four purposes of an uncertainty design have the same structure and differ in what
the levels should be: plausible bounds for sensitivity analysis and situational
awareness, extremes for horizon scanning, and for forecasting a set of levels that
covers the uncertainty **with probabilities** (`weights` on every axis), so that the
weighted scenarios make one forecast ([`scenario_forecast`](@ref)).

`purpose` states what the design is for and `audience` whom it is for; a purpose that
the axes cannot serve throws an `ArgumentError` that says why. They are part of the
design because the same scenarios read differently for different audiences, and are
hard to recover afterwards.

The design is the plan of a table of runs: [`design_actions`](@ref) and
[`design_draws`](@ref) are what [`outcomes`](@ref) takes, with replicates paired across
scenarios by its common random numbers, and `DecisionProblem(design, table, utility)`
is the problem the runs answer. See [`decision_robustness`](@ref) and
[`factor_effects`](@ref).
"""
struct ScenarioDesign
    axes::Vector{ScenarioAxis}
    purpose::Union{Nothing,Symbol}
    audience::String
    cells::Vector{Vector{Int}}
    function ScenarioDesign(axes::Vector{<:ScenarioAxis}, purpose, audience,
                            cells::Vector{Vector{Int}})
        isempty(axes) && throw(ArgumentError("a design needs at least one axis"))
        allunique(a.name for a in axes) ||
            throw(ArgumentError("the axes of a design must have distinct names"))
        isempty(cells) && throw(ArgumentError("the design has no scenario"))
        design = new(collect(ScenarioAxis, axes), purpose, String(audience), cells)
        purpose === nothing || _check_purpose(design, purpose)
        return design
    end
end
function ScenarioDesign(axes::ScenarioAxis...; purpose::Union{Nothing,Symbol}=nothing,
                        audience::AbstractString="")
    list = collect(ScenarioAxis, axes)
    isempty(list) && throw(ArgumentError("a design needs at least one axis"))
    cells = vec([collect(c)
                 for c in Iterators.product((eachindex(a.levels) for a in list)...)])
    return ScenarioDesign(list, purpose, audience, cells)
end

"""
    design_class(design::ScenarioDesign) -> Symbol

The class of a design by the kinds of its axes: `:decision` (interventions only),
`:value_of_information` (at least one intervention and one uncertainty) or
`:uncertainty` (uncertainties only). See [`ScenarioDesign`](@ref).
"""
function design_class(design::ScenarioDesign)
    decisions = any(a -> a.kind === :decision, design.axes)
    uncertainties = any(a -> a.kind === :uncertainty, design.axes)
    return decisions && uncertainties ? :value_of_information :
           decisions ? :decision : :uncertainty
end

function _check_purpose(design::ScenarioDesign, purpose::Symbol)
    purpose in _DESIGN_PURPOSES ||
        throw(ArgumentError("purpose must be one of $(_DESIGN_PURPOSES), got :$purpose"))
    class = design_class(design)
    wanted = purpose === :decision_making ? :decision :
             purpose === :value_of_information ? :value_of_information : :uncertainty
    reason = wanted === :decision ?
             "compares interventions, so every axis must be a " *
             "decision; with an uncertainty axis the design is " *
             "one for the value of information" :
             wanted === :value_of_information ?
             "asks whether the better intervention depends on an uncertainty, so it " *
             "needs at least one decision axis and one uncertainty axis" :
             "explores uncertainty, so every axis must be an uncertainty; with a " *
             "decision axis the design is one for the value of information"
    class === wanted ||
        throw(ArgumentError("a design for :$purpose $reason; this one has the axes " *
                            join(("$(a.name) ($(a.kind))" for a in design.axes), ", ")))
    if purpose === :forecasting
        missing_weights = [a.name for a in design.axes if a.weights === nothing]
        isempty(missing_weights) ||
            throw(ArgumentError("a forecasting design needs probabilities of the levels " *
                                "of every axis, so that the scenarios combine into one " *
                                "forecast; no weights on $(join(missing_weights, ", "))"))
    end
    return nothing
end

_cell_label(axes, cell) = Symbol(join((a.levels[i] for (a, i) in zip(axes, cell)), "_"))

function _axis_cells(design::ScenarioDesign, kind::Symbol)
    which = findall(a -> a.kind === kind, design.axes)
    axes = design.axes[which]
    cells = unique([cell[which] for cell in design.cells])
    return axes, cells
end

"""
    scenarios(design::ScenarioDesign) -> Vector{NamedTuple}

The scenarios of a design, one `(; scenario, <axis> = level, ...)` per scenario, in
order (the first axis varies fastest). `scenario` is the label, the levels joined by
`_`. A Tables.jl row table.
"""
function scenarios(design::ScenarioDesign)
    names = (:scenario, (a.name for a in design.axes)...)
    return [NamedTuple{names}((_cell_label(design.axes, cell),
                               (a.levels[i] for (a, i) in zip(design.axes, cell))...))
            for cell in design.cells]
end

"""
    design_actions(design::ScenarioDesign) -> Vector{Pair{Symbol,NamedTuple}}

The actions of a design, for [`outcomes`](@ref): one `label => (; axis = value, ...)`
per combination of the levels of its decision axes. A model package's `apply_action`
receives the named tuple. A design with no decision axis has the single action
`:projection => (;)`.
"""
function design_actions(design::ScenarioDesign)
    axes, cells = _axis_cells(design, :decision)
    isempty(axes) && return [:projection => (;)]
    names = Tuple(a.name for a in axes)
    return [_cell_label(axes, cell) => NamedTuple{names}(Tuple(a.values[i]
                                                               for (a, i) in
                                                                   zip(axes, cell)))
            for cell in cells]
end

function _state_weights(axes, cells)
    any(a -> a.weights === nothing, axes) && return nothing
    return [prod(a.weights[i] for (a, i) in zip(axes, cell); init=1 // 1) for cell in cells]
end

"""
    design_draws(design::ScenarioDesign) -> SampleBelief

The states of a design as draws, for [`outcomes`](@ref): one draw per combination of
the levels of its uncertainty axes, labelled by the levels, with the axes' values as
its parameters (so `with_parameters` receives `(; axis = value, ...)`). The weights
are the products of the axes' weights when every uncertainty axis has them, and
equal otherwise, **which is then a convenience for running the model and not a
belief**: use [`design_belief`](@ref) for the analysis. A design with no uncertainty
axis has one draw, `:all`.
"""
function design_draws(design::ScenarioDesign)
    axes, cells = _axis_cells(design, :uncertainty)
    isempty(axes) && return SampleBelief(1; labels=[:all])
    names = Tuple(a.name for a in axes)
    columns = NamedTuple{names}(Tuple([a.values[cell[k]] for cell in cells]
                                      for (k, a) in enumerate(axes)))
    columns = map(c -> collect(promote_type(typeof.(c)...), c), columns)
    return SampleBelief(columns; labels=[_cell_label(axes, cell) for cell in cells],
                        weights=_state_weights(axes, cells))
end

"""
    design_belief(design::ScenarioDesign)

What a design says about its states. A [`ScenarioSet`](@ref) when some uncertainty
axis has no probabilities: the scenarios are held possible and nothing more, so
state-wise rules and regret apply and an expectation over them does not. A
[`FiniteBelief`](@ref) with the product of the axes' weights (the axes taken to be
independent) when every uncertainty axis has them.
"""
function design_belief(design::ScenarioDesign)
    axes, cells = _axis_cells(design, :uncertainty)
    isempty(axes) && return FiniteBelief([:all], [1 // 1])
    labels = [_cell_label(axes, cell) for cell in cells]
    weights = _state_weights(axes, cells)
    return weights === nothing ? ScenarioSet(labels) : FiniteBelief(labels, weights)
end

"""
    DecisionProblem(design::ScenarioDesign, table::OutcomeTable, utility; baseline=nothing)

The decision problem that the runs of a design answer: the payoff table of
[`payoff_table`](@ref)`(table, utility)`, whose actions and draws must be those of
[`design_actions`](@ref) and [`design_draws`](@ref), with the belief of
[`design_belief`](@ref). Without probabilities on the uncertainty axes the problem is
over a scenario set: [`decide`](@ref) with [`Maximin`](@ref) or
[`MinimaxRegret`](@ref), and [`decision_robustness`](@ref), apply; expected utility and
[`evpi`](@ref) need the probabilities. A fraction of a design that leaves some action
without some state cannot make a payoff table.
"""
function DecisionProblem(design::ScenarioDesign, table::OutcomeTable, utility;
                         baseline=nothing)
    actions = first.(design_actions(design))
    states = state_labels(design_draws(design))
    (sort(action_labels(table)) == sort(actions) &&
     sort(state_labels(table)) == sort(states)) ||
        throw(ArgumentError("the table has the actions $(action_labels(table)) and the " *
                            "draws $(state_labels(table)); the design has $actions and " *
                            "$states"))
    length(design.cells) == length(actions) * length(states) ||
        throw(ArgumentError("the design is a fraction that leaves some action without " *
                            "some state, so it has no payoff table"))
    payoff = payoff_table(table, utility)
    belief = design_belief(design)
    order = [findfirst(==(s), state_labels(belief)) for s in payoff.states]
    belief = belief isa ScenarioSet ? ScenarioSet(payoff.states) :
             FiniteBelief(payoff.states, belief.probabilities[order])
    return DecisionProblem(payoff, belief; baseline)
end

"""
    fractional_design(design::ScenarioDesign, generators...) -> ScenarioDesign

A **fractional factorial**: the part of a design whose axes all have two levels in
which each generated axis is tied to others. A generator `:c => (:a, :b)` keeps the
scenarios where the level of `c` is the product of the levels of `a` and `b`, with the
first level of an axis counted as −1 and the second as +1; `:c => (:a,)` ties `c` to
`a`. One generator halves the design. The main effect of every axis can still be
estimated ([`factor_effects`](@ref)), at the price that the generated axis is
**aliased** with the interaction of the axes that generate it: the two cannot be told
apart. This is how many uncertainties can be explored with few scenarios, each of
which a model (or every model of a hub) must run.
"""
function fractional_design(design::ScenarioDesign, generators::Pair...)
    isempty(generators) && throw(ArgumentError("give at least one generator"))
    all(a -> length(a.levels) == 2, design.axes) ||
        throw(ArgumentError("a fractional factorial needs two levels on every axis"))
    position(name) = something(findfirst(a -> a.name === Symbol(name), design.axes),
                               0)
    cells = design.cells
    for (generated, parents) in generators
        list = parents isa Union{Symbol,AbstractString} ? (parents,) : parents
        g = position(generated)
        ps = [position(p) for p in list]
        (g > 0 && all(>(0), ps) && !(g in ps) && !isempty(ps)) ||
            throw(ArgumentError("the generator $generated => $parents must name axes " *
                                "of the design, the generated one not among its parents"))
        cells = [c for c in cells if 2c[g] - 3 == prod(2c[p] - 3 for p in ps)]
    end
    return ScenarioDesign(design.axes, design.purpose, design.audience, cells)
end

function _scenario_values(design::ScenarioDesign, values)
    labels = [_cell_label(design.axes, cell) for cell in design.cells]
    if values isa AbstractDict || values isa NamedTuple
        function get_value(l)
            return values isa NamedTuple ?
                   (haskey(values, l) ? values[l] : nothing) :
                   get(values, l, get(values, String(l), nothing))
        end
        out = [get_value(l) for l in labels]
        any(isnothing, out) &&
            throw(ArgumentError("no value was given for the scenarios " *
                                "$(labels[findall(isnothing, out)])"))
        return labels, out
    end
    length(values) == length(labels) ||
        throw(ArgumentError("$(length(values)) values for the $(length(labels)) " *
                            "scenarios of the design"))
    return labels, collect(values)
end

"""
    factor_effects(design::ScenarioDesign, values) -> NamedTuple

How an output differs along the axes of a design: the sensitivity analysis that a
scenario design is. `values` has one number per scenario, the projected output (a
mean over replicates, say), in the order of [`scenarios`](@ref), or keyed by the
scenario labels.

The result is `(; grand_mean, total, main, interactions)`. `main` has one
`(; axis, kind, levels, means, effect, share)` per axis: the mean of the output at each
of its levels, `effect` the largest minus the smallest of those means (the second
level minus the first for a two-level axis), and `share` its part of the sum of
squares of the output about its mean. `interactions` has one `(; axes, share)` per
pair of axes, for a full factorial: the part that their main effects do not account
for and their joint means do. With two axes the shares of the two main effects and
their interaction add up to one. In a fraction ([`fractional_design`](@ref)) main
effects are aliased with interactions, and `interactions` is empty.

Every scenario counts equally: the shares describe the design, not a belief about
which scenario is likely. For a share weighted by a belief, see
[`variance_reduction_table`](@ref).
"""
function factor_effects(design::ScenarioDesign, values)
    _, raw = _scenario_values(design, values)
    all(v -> v isa Real, raw) ||
        throw(ArgumentError("factor_effects needs one number per scenario"))
    y = Float64.(raw)
    n = length(y)
    grand = sum(y) / n
    total = sum(abs2, y .- grand)
    function level_means(k)
        return [(members = findall(c -> c[k] == l, design.cells);
                 isempty(members) ? NaN : sum(y[members]) / length(members))
                for l in eachindex(design.axes[k].levels)]
    end
    function explained(k)
        return sum(count(c -> c[k] == l, design.cells) * (m - grand)^2
                   for (l, m) in enumerate(level_means(k)) if !isnan(m))
    end
    share(x) = iszero(total) ? 0.0 : x / total
    main = map(enumerate(design.axes)) do (k, axis)
        means = level_means(k)
        effect = length(means) == 2 ? means[2] - means[1] :
                 maximum(filter(!isnan, means)) - minimum(filter(!isnan, means))
        return (; axis=axis.name, kind=axis.kind, levels=axis.levels, means, effect,
                share=share(explained(k)))
    end
    full = n == prod(length(a.levels) for a in design.axes)
    interactions = NamedTuple[]
    if full
        for i in eachindex(design.axes), j in (i + 1):length(design.axes)
            joint = 0.0
            for a in eachindex(design.axes[i].levels), b in eachindex(design.axes[j].levels)
                members = findall(c -> c[i] == a && c[j] == b, design.cells)
                joint += length(members) * (sum(y[members]) / length(members) - grand)^2
            end
            push!(interactions,
                  (; axes=(design.axes[i].name, design.axes[j].name),
                   share=share(max(joint - explained(i) - explained(j), 0.0))))
        end
    end
    return (; grand_mean=grand, total, main, interactions)
end

"""
    scenario_forecast(design::ScenarioDesign, projections;
                      probabilities=[0.025, 0.25, 0.5, 0.75, 0.975]) -> NamedTuple

One forecast from the scenarios of a **forecasting design**: the mixture of the
scenarios' projections with the scenarios' probabilities (the products of the axes'
weights). `projections` has, per scenario (in the order of [`scenarios`](@ref) or
keyed by label), either a vector of replicate projections of one output, or a single
number, its mean.

The result is `(; mean, variance, between, within, weights, quantiles)`: the mean and
variance of the mixture, the variance split into the part **between** scenarios (what
the scenario axes are uncertain about) and the part **within** them (the rest, as far
as the replicates express it; zero for single numbers), and the quantiles of the
mixture at `probabilities` (`nothing` without replicates). A design whose axes have no
probabilities is a set of conditional projections and has no forecast: that is the
difference between the two, and this function throws.
"""
function scenario_forecast(design::ScenarioDesign, projections;
                           probabilities::AbstractVector{<:Real}=[0.025, 0.25, 0.5, 0.75,
                                                                  0.975])
    design_class(design) === :uncertainty ||
        throw(ArgumentError("a forecast combines scenarios that differ in uncertainties " *
                            "only; this design has a decision axis, and its scenarios " *
                            "are compared, not combined"))
    weights = _state_weights(design.axes, design.cells)
    weights === nothing &&
        throw(ArgumentError("the axes have no probabilities, so the scenarios are " *
                            "conditional projections and do not combine into a " *
                            "forecast; give `weights` to every uncertainty axis"))
    w = Float64.(weights)
    w ./= sum(w)
    labels, raw = _scenario_values(design, projections)
    replicated = all(v -> v isa AbstractVector{<:Real}, raw)
    replicated || all(v -> v isa Real, raw) ||
        throw(ArgumentError("give every scenario a number, or every scenario a vector " *
                            "of replicates"))
    means = replicated ? [sum(v) / length(v) for v in raw] : Float64.(raw)
    within_each = replicated ?
                  [sum(abs2, v .- m) / length(v) for (v, m) in zip(raw, means)] :
                  zeros(length(means))
    centre = dot(w, means)
    between = dot(w, (means .- centre) .^ 2)
    within = dot(w, within_each)
    quantiles = nothing
    if replicated
        all(p -> 0 <= p <= 1, probabilities) ||
            throw(ArgumentError("probabilities must lie in [0, 1]"))
        pooled = reduce(vcat, [Float64.(v) for v in raw])
        mass = reduce(vcat,
                      [fill(w[k] / length(v), length(v)) for (k, v) in enumerate(raw)])
        quantiles = _weighted_quantiles(pooled, mass, probabilities)
    end
    return (; mean=centre, variance=between + within, between, within,
            weights=NamedTuple{Tuple(labels)}(Tuple(w)), quantiles)
end

"""
    decision_robustness(problem::DecisionProblem) -> NamedTuple

Whether the better action depends on the state: the reading of a **value of
information design** (Runge et al. 2024), for a problem over a few states under any
belief form. The result is `(; best, robust, reversal, losses)`:

- `best`: for every state, the feasible actions with the largest payoff there;
- `robust`: the actions that are best in every state (often none);
- `reversal`: `true` when no action is best in every state, so that the preferred
  action reverses somewhere along the uncertainty: knowing the state would change
  what is done, and resolving the uncertainty has value;
- `losses`: for every action, its largest regret over the states, what is lost at
  worst by committing to it now.

`reversal = false` says that the choice is robust to the uncertainties **that the
design explored**, which is a comfort only if those were the ones that matter; a
design in which no uncertainty ever reverses the choice may have the wrong
uncertainty axes. With probabilities on the states the size of what is at stake is
[`evpi`](@ref), which is positive exactly when the reversal has positive probability.
"""
function decision_robustness(problem::DecisionProblem)
    Q = _feasible_Q(problem)
    actions = feasible_actions(problem)
    states = problem.payoff.states
    tops = [maximum(view(Q, :, j)) for j in axes(Q, 2)]
    best = [states[j] => actions[findall(==(tops[j]), view(Q, :, j))] for j in axes(Q, 2)]
    robust = [a
              for (k, a) in enumerate(actions)
              if all(Q[k, j] == tops[j] for j in axes(Q, 2))]
    losses = [a => maximum(tops[j] - Q[k, j] for j in axes(Q, 2))
              for (k, a) in enumerate(actions)]
    return (; best, robust, reversal=isempty(robust), losses)
end
