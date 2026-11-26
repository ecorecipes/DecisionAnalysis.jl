# The outcome table: what a model produced, by draw × action × replicate × named outcome,
# before any utility is applied. Layout of `values`: [action, draw, replicate, outcome],
# so that a slice over replicates is a cell of the payoff table Q[action, draw].

const _OK = "ok"
const _ABSENT = "absent"

# The distinct labels of a column in order of first appearance, and the position of each
# of its `n` entries among them.
function _index_labels(col, n::Integer)
    labels = Symbol[]
    seen = Dict{Symbol,Int}()
    idx = Vector{Int}(undef, n)
    for i in 1:n
        l = _label(col[i])
        idx[i] = get!(seen, l) do
            push!(labels, l)
            return length(labels)
        end
    end
    return labels, idx
end

"""
    OutcomeTable(runs; draw=:draw, action=:action, replicate=:replicate, outcomes,
                 status=nothing, ok=("ok",), parameters=nothing, key=draw,
                 weights=nothing, actions=nothing)

The outcomes of a model over **draws × actions × replicates × named outcomes**, built
from `runs`, any Tables.jl table with one row per run (a vector of named tuples, a
data frame, a CSV file read into a table). The keywords name its columns:

- `draw`, `action`, `replicate`: the columns that identify the parameter draw, the
  action and the replicate of each run. Their values become labels (`Symbol` of
  their text), in order of first appearance. A run that appears twice throws
  [`InvalidOutcomeError`](@ref).
- `outcomes`: the names of the outcome columns to keep (numbers).
- `status`: the name of a column that says whether a run succeeded, and `ok` the
  values that count as success. **A run with any other status is kept and counted,
  never silently dropped**: it is left out of every mean and standard error, and it
  is listed by [`failed_runs`](@ref) and counted by [`run_counts`](@ref). Without
  `status` every run counts as a success, and its outcomes must be finite.
- `parameters`: the parameter values of the draws. Either a Tables.jl table with one
  row per draw and a column named by `key` (default: the name given as `draw`) that
  identifies it, or a list of column names of `runs` itself (each constant within a
  draw). With a table, the draws take its row order, and every draw of `runs` must be
  in it.
- `weights`: the weights of the draws: `nothing` (equal), a vector in draw order, or
  the name of a column of the `parameters` table. They are normalised as for a
  [`SampleBelief`](@ref).
- `actions`: the order of the actions, when the order of first appearance is not the
  one wanted (a permutation of the labels found).

Cells need not have the same number of runs: replicates that are absent or failed
are simply not averaged. A cell with no usable run is allowed in the table, and
[`payoff_table`](@ref) refuses to make a payoff from it.

The table is the seam with model packages: a model package writes a table of runs
(or implements the adapter verbs for [`outcomes`](@ref)) and knows nothing of this
package. A utility turns it into a payoff table: [`payoff_table`](@ref).

Accessors: [`state_labels`](@ref) (the draws), [`action_labels`](@ref),
[`outcome_names`](@ref), [`parameter_draws`](@ref), [`draw_weights`](@ref),
[`replicate_counts`](@ref), [`run_counts`](@ref), [`failed_runs`](@ref), and
`SampleBelief(table)`.

```jldoctest
julia> runs = [(; draw=d, action=a, replicate=r, frogs=10d + (a == "fence") + r)
               for d in 1:2 for a in ("none", "fence") for r in 1:3];

julia> t = OutcomeTable(runs; outcomes=[:frogs]);

julia> run_counts(t)
(runs = 12, ok = 12, failed = 0, absent = 0)

julia> payoff_table(t, IndividualsUtility(:frogs)).Q
2×2 Matrix{Rational{Int64}}:
 12  22
 13  23
```
"""
struct OutcomeTable{T<:Real,C<:NamedTuple,W<:Real}
    draws::Vector{Symbol}
    actions::Vector{Symbol}
    outcomes::Vector{Symbol}
    replicates::Vector{Symbol}
    values::Array{T,4}
    status::Array{String,3}
    parameters::C
    weights::Vector{W}
    function OutcomeTable(draws::Vector{Symbol}, actions::Vector{Symbol},
                          outcomes::Vector{Symbol}, replicates::Vector{Symbol},
                          values::Array{T,4}, status::Array{String,3}, parameters::C,
                          weights::Vector{W}) where {T<:Real,C<:NamedTuple,W<:Real}
        size(values) ==
        (length(actions), length(draws), length(replicates), length(outcomes)) ||
            throw(InvalidOutcomeError("the outcome array has size $(size(values)) for " *
                                      "$(length(actions)) actions, $(length(draws)) " *
                                      "draws, $(length(replicates)) replicates and " *
                                      "$(length(outcomes)) outcomes"))
        size(status) == size(values)[1:3] ||
            throw(InvalidOutcomeError("the status array does not match the outcome array"))
        for (what, labels) in (("draw", draws), ("action", actions),
                               ("outcome", outcomes), ("replicate", replicates))
            isempty(labels) && throw(InvalidOutcomeError("the table has no $what"))
            allunique(labels) ||
                throw(InvalidOutcomeError("the $what labels must be distinct"))
        end
        # Reuses the checks of a sample belief on the parameters and weights.
        SampleBelief(draws, parameters, weights)
        return new{T,C,W}(draws, actions, outcomes, replicates, values, status, parameters,
                          weights)
    end
end

_label(x) = Symbol(string(x))

function OutcomeTable(runs; draw=:draw, action=:action, replicate=:replicate, outcomes,
                      status=nothing, ok=(_OK,), parameters=nothing, key=draw,
                      weights=nothing, actions=nothing)
    Tables.istable(runs) ||
        throw(ArgumentError("expected a Tables.jl table of runs, got $(typeof(runs))"))
    columns = Tables.columns(runs)
    available = collect(Symbol, Tables.columnnames(columns))
    names = _labels(outcomes)
    isempty(names) && throw(ArgumentError("name at least one outcome column"))
    from_runs = parameters isa Union{Tuple,AbstractVector} &&
                all(x -> x isa Union{Symbol,AbstractString}, parameters)
    carried = from_runs ? _labels(parameters) : Symbol[]
    wanted = vcat(Symbol(draw), Symbol(action), Symbol(replicate), names, carried,
                  status === nothing ? Symbol[] : [Symbol(status)])
    for c in wanted
        c in available ||
            throw(ArgumentError("the table of runs has no column $c; columns: $available"))
    end
    column(c) = Tables.getcolumn(columns, Symbol(c))
    dcol, acol, rcol = column(draw), column(action), column(replicate)
    scol = status === nothing ? nothing : column(status)
    ocols = [column(c) for c in names]
    okset = Set(string(x) for x in (ok isa Union{AbstractString,Symbol} ? (ok,) : ok))
    n = length(dcol)
    n >= 1 || throw(InvalidOutcomeError("the table of runs is empty"))

    dlabels, di = _index_labels(dcol, n)
    alabels, ai = _index_labels(acol, n)
    rlabels, ri = _index_labels(rcol, n)

    if actions !== nothing
        order = _labels(actions)
        sort(order) == sort(alabels) ||
            throw(ArgumentError("`actions` must list the actions of the runs, " *
                                "$alabels, in the order wanted; got $order"))
        perm = [findfirst(==(a), order) for a in alabels]
        ai = perm[ai]
        alabels = order
    end

    # Parameters and weights of the draws, and the draw order.
    pars = (;)
    w = weights
    if from_runs && !isempty(carried)
        first_row = zeros(Int, length(dlabels))
        for i in 1:n
            first_row[di[i]] == 0 && (first_row[di[i]] = i)
        end
        for c in carried
            col = column(c)
            for i in 1:n
                isequal(col[i], col[first_row[di[i]]]) ||
                    throw(InvalidOutcomeError("parameter column $c is not constant " *
                                              "within draw $(dlabels[di[i]])"))
            end
        end
        pars = NamedTuple{Tuple(carried)}(Tuple([column(c)[i] for i in first_row]
                                                for c in carried))
    elseif parameters !== nothing && !from_runs
        ptable = _column_table(parameters)
        haskey(ptable, Symbol(key)) ||
            throw(ArgumentError("the parameter table has no column $key to identify " *
                                "the draws; columns: $(collect(keys(ptable)))"))
        plabels = _label.(ptable[Symbol(key)])
        allunique(plabels) ||
            throw(InvalidOutcomeError("the parameter table lists a draw twice"))
        missing_draws = setdiff(dlabels, plabels)
        isempty(missing_draws) ||
            throw(InvalidOutcomeError("draws $(missing_draws) of the runs are not in the " *
                                      "parameter table"))
        place = Dict(l => k for (k, l) in enumerate(plabels))
        di = getindex.(Ref(place), dlabels[di])
        dlabels = plabels
        pars = Base.structdiff(ptable, NamedTuple{(Symbol(key),)})
        if weights isa Symbol
            haskey(pars, weights) ||
                throw(ArgumentError("the parameter table has no column $weights to " *
                                    "use as weights"))
            w = pars[weights]
            pars = Base.structdiff(pars, NamedTuple{(weights,)})
        end
    end
    w isa Symbol &&
        throw(ArgumentError("`weights = :$w` names a column of a `parameters` table, " *
                            "and none was given"))
    nd = length(dlabels)
    wv = w === nothing ? fill(1 // nd, nd) : _normalise_weights(collect(w))
    length(wv) == nd || throw(ArgumentError("$(length(wv)) weights for $nd draws"))

    state = fill(_ABSENT, length(alabels), nd, length(rlabels))
    raw = Array{Any}(undef, length(alabels), nd, length(rlabels), length(names))
    fill!(raw, 0)
    T = Union{}
    for i in 1:n
        a, d, r = ai[i], di[i], ri[i]
        state[a, d, r] == _ABSENT ||
            throw(InvalidOutcomeError("the run of action $(alabels[a]), draw " *
                                      "$(dlabels[d]), replicate $(rlabels[r]) appears " *
                                      "more than once"))
        s = scol === nothing ? _OK : string(scol[i])
        if scol !== nothing && !(s in okset)
            state[a, d, r] = (s == _OK || s == _ABSENT) ? "failed:" * s : s
            continue
        end
        state[a, d, r] = _OK
        for (k, col) in enumerate(ocols)
            v = col[i]
            (v isa Real && isfinite(v)) ||
                throw(InvalidOutcomeError("outcome $(names[k]) of action " *
                                          "$(alabels[a]), draw $(dlabels[d]), replicate " *
                                          "$(rlabels[r]) is $(repr(v)), not a finite " *
                                          "number, in a run whose status is acceptable"))
            raw[a, d, r, k] = v
            T = promote_type(T, typeof(v))
        end
    end
    T === Union{} && throw(InvalidOutcomeError("no run of the table has an acceptable " *
                                               "status"))
    T === Bool && (T = Int)
    return OutcomeTable(dlabels, alabels, names, rlabels, Array{T,4}(raw), state, pars, wv)
end

state_labels(t::OutcomeTable) = t.draws
action_labels(t::OutcomeTable) = t.actions
parameter_draws(t::OutcomeTable) = t.parameters
draw_weights(t::OutcomeTable) = t.weights

"""
    SampleBelief(t::OutcomeTable)

The sample belief of an outcome table: its draws, their parameters and their
weights, with the draws as states.
"""
SampleBelief(t::OutcomeTable) = SampleBelief(t.draws, t.parameters, t.weights)

"""
    outcome_names(t::OutcomeTable) -> Vector{Symbol}

The names of the outcomes recorded in the table.
"""
outcome_names(t::OutcomeTable) = t.outcomes

"""
    replicate_counts(t::OutcomeTable) -> Matrix{Int}

The number of usable runs in every cell, `[action, draw]`: the runs that are present
and whose status is acceptable. These are the runs that a payoff averages.
"""
function replicate_counts(t::OutcomeTable)
    return [count(==(_OK), view(t.status, a, d, :))
            for a in eachindex(t.actions), d in eachindex(t.draws)]
end

"""
    single_replicate(t::OutcomeTable) -> Bool

Whether some action and draw of the table have exactly one usable run
([`replicate_counts`](@ref)), so that the payoff table made from it would be flagged
(`single_replicate(payoff_table(t, utility))`): such a cell is one outcome of the
model, not a conditional expectation, and has no standard error.
"""
single_replicate(t::OutcomeTable) = any(==(1), replicate_counts(t))

"""
    run_counts(t::OutcomeTable) -> NamedTuple

How the runs of the table divide: `(; runs, ok, failed, absent)`. `runs` is the
number of rows the table was built from, `ok` those with an acceptable status,
`failed` those with any other status (kept, counted, and left out of every mean),
and `absent` the number of positions of the draws × actions × replicates grid for
which there was no row.
"""
function run_counts(t::OutcomeTable)
    ok = count(==(_OK), t.status)
    absent = count(==(_ABSENT), t.status)
    return (; runs=length(t.status) - absent, ok, failed=length(t.status) - absent - ok,
            absent)
end

"""
    failed_runs(t::OutcomeTable) -> Vector{NamedTuple}

The runs whose status was not acceptable, one `(; action, draw, replicate, status)`
per run, in table order.
"""
function failed_runs(t::OutcomeTable)
    return [(; action=t.actions[a], draw=t.draws[d], replicate=t.replicates[r],
             status=t.status[a, d, r])
            for a in eachindex(t.actions), d in eachindex(t.draws),
            r in eachindex(t.replicates)
            if t.status[a, d, r] != _OK && t.status[a, d, r] != _ABSENT]
end

"""
    combine_actions(tables::OutcomeTable...) -> OutcomeTable

One outcome table with the actions of several, for runs of the **same draws and
replicates** made at different times: a stored table and the runs of new actions.
The tables must have the same draws (labels and weights) and the same replicates
(labels), in any order, and no action in common; the parameters they both carry
must agree. The result has the draws, replicates, parameters and weights of the
first table, the actions of all of them in order, and the outcomes that every table
records, in the order of the first. Failed and absent runs keep their status.

Nothing here can check that replicate `r` of a draw used the same random numbers in
every table. It did when the runs were seeded by one rule: the same `seed`, or the
same `run_seed`, in every call of [`outcomes`](@ref), or a `run_seed` that is the
rule by which the stored runs were made. Only then is the difference between two
actions of the combined table free of the noise that common random numbers remove,
and [`crn_correlation`](@ref) reports what was achieved.

```jldoctest
julia> runs(a, shift) = [(; draw=d, action=a, replicate=r, frogs=10d + r + shift)
                         for d in 1:2 for r in 1:3];

julia> stored = OutcomeTable(runs("none", 0); outcomes=[:frogs]);

julia> t = combine_actions(stored, OutcomeTable(runs("fence", 2); outcomes=[:frogs]));

julia> action_labels(t), run_counts(t).ok
([:none, :fence], 12)
```
"""
function combine_actions(first::OutcomeTable, rest::OutcomeTable...)
    status, actions = first.status, copy(first.actions)
    names = filter(o -> all(t -> o in t.outcomes, rest), first.outcomes)
    isempty(names) && throw(ArgumentError("the tables record no outcome in common"))
    pick(t) = t.values[:, :, :, [findfirst(==(o), t.outcomes) for o in names]]
    values = pick(first)
    for (k, t) in enumerate(rest)
        which = "table $(k + 1)"
        sort(t.draws) == sort(first.draws) ||
            throw(ArgumentError("$which does not have the draws of the first: " *
                                "$(symdiff(t.draws, first.draws)) are in one only"))
        sort(t.replicates) == sort(first.replicates) ||
            throw(ArgumentError("$which does not have the replicates of the first " *
                                "($(first.replicates)): it has $(t.replicates)"))
        shared = intersect(actions, t.actions)
        isempty(shared) ||
            throw(ArgumentError("$which repeats the actions $shared; drop them from " *
                                "one table, since a run cannot be in the table twice"))
        d = [findfirst(==(l), t.draws) for l in first.draws]
        r = [findfirst(==(l), t.replicates) for l in first.replicates]
        t.weights[d] == first.weights ||
            throw(ArgumentError("$which gives the draws other weights than the first"))
        for name in intersect(keys(first.parameters), keys(t.parameters))
            isequal(t.parameters[name][d], first.parameters[name]) ||
                throw(ArgumentError("$which has other values of the parameter $name " *
                                    "than the first, so its draws are not the same"))
        end
        values = cat(values, pick(t)[:, d, r, :]; dims=1)
        status = cat(status, t.status[:, d, r]; dims=1)
        append!(actions, t.actions)
    end
    return OutcomeTable(copy(first.draws), actions, names, copy(first.replicates), values,
                        status, first.parameters, copy(first.weights))
end

function Base.show(io::IO, t::OutcomeTable)
    c = run_counts(t)
    return print(io, "OutcomeTable(", length(t.draws), " draws × ", length(t.actions),
                 " actions × ", length(t.replicates), " replicates; outcomes ",
                 join(t.outcomes, ", "), "; ", c.ok, " runs ok, ", c.failed, " failed, ",
                 c.absent, " absent)")
end

# The driver
# ----------

# SplitMix64 finaliser: a fixed integer hash, the same on every Julia version.
function _splitmix64(x::UInt64)
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end

# The seed of replicate `r` at draw `d`: the same for every action.
function _run_seed(seed::Integer, d::Integer, r::Integer)
    return _splitmix64(_splitmix64(_splitmix64(seed % UInt64) + UInt64(d)) + UInt64(r))
end

_seed_of_run(seed::Integer, ::Nothing, θ, d::Integer, r::Integer) = _run_seed(seed, d, r)
function _seed_of_run(::Nothing, run_seed, θ, d::Integer, r::Integer)
    s = run_seed(θ, r)
    s isa Integer ||
        throw(ArgumentError("`run_seed(θ, r)` must return an integer seed, got " *
                            "$(typeof(s)) at draw $d, replicate $r"))
    return s
end

_named_actions(actions::NamedTuple) = [k => v for (k, v) in pairs(actions)]
_named_actions(actions::AbstractDict) = [Symbol(k) => v for (k, v) in actions]
function _named_actions(actions)
    return [a isa Pair ? Symbol(first(a)) => last(a) : Symbol(string(a)) => a
            for a in actions]
end

_outcome_tuple(o::NamedTuple) = o
_outcome_tuple(o::Real) = (; outcome=o)
function _outcome_tuple(o)
    throw(InvalidOutcomeError("`outcome` must return a named tuple of numbers, got " *
                              "$(typeof(o))"))
end

"""
    outcomes(prob, actions, draws; replicates=1, seed, run_seed=nothing,
             on_error=:record) -> OutcomeTable

Run a model over draws × actions × replicates and collect the named outcomes. The
model is reached through the adapter verbs only, which must have methods for the
type of `prob`: each run is

    p = apply_action(with_parameters(prob, θ), a)
    outcome(p, simulate(p; rng))

([`with_parameters`](@ref), [`apply_action`](@ref), [`simulate`](@ref),
[`outcome`](@ref); `outcome(p, solution)` falls back on `outcome(solution)`).

- `actions`: the actions, as `label => action` pairs, a named tuple, a dictionary, or
  a list of values that are their own labels. Include no intervention.
- `draws`: a [`SampleBelief`](@ref), or a Tables.jl table of parameter draws (equal
  weights). Each `θ` is one row, a named tuple. Draws are propagated one by one; a
  mean of the parameters is never plugged in.
- `replicates`: runs per cell. A stochastic model needs several, so that a cell of
  the payoff table is a conditional expectation and has a standard error; a table
  with one is flagged ([`single_replicate`](@ref)).
- `seed`: an integer, required unless `run_seed` is given. **Common random
  numbers**: the run of replicate `r` at draw `d` gets a `Random.Xoshiro` whose seed
  is a fixed integer hash of `(seed, d, r)`, the same for every action, so the
  difference between two actions in a cell is not blurred by different random
  numbers. How long two runs stay comparable after an intervention is up to the
  model (see [`simulate`](@ref)).
- `run_seed`: the model's own seed convention, in place of `seed`: a function
  `run_seed(θ, r)` of the draw (its row of parameters, a named tuple) and the
  replicate number `r` (from 1) that returns the integer seed of that run's
  `Xoshiro`. It is called once per draw and replicate, so the random numbers are
  still common to the actions. Use it when runs of the same model made elsewhere (a
  stored table, an earlier call) seeded their generators by a rule of their own, so
  that new runs share random numbers with them and the two can be put in one table
  ([`combine_actions`](@ref)): an action then differs from a stored one by the
  action alone. The function must
  be deterministic and should give different seeds to different runs; two runs with
  the same seed throw an `ArgumentError`, because their outcomes would be copies and
  not replicates. Give `seed` or `run_seed`, not both.

A run that throws is kept as a failed run with the status `error:<exception type>`
(see [`failed_runs`](@ref)); an `InterruptException` is rethrown. If every run fails,
[`InvalidOutcomeError`](@ref) is thrown with the first error's message. With
`on_error=:throw` the first exception of any run is rethrown instead, with its own
message and stack trace: use it while writing an adapter or a menu of actions, when a
failed run is a mistake to find and not an outcome of the model to count.
"""
function outcomes(prob, actions, draws; replicates::Integer=1,
                  seed::Union{Nothing,Integer}=nothing, run_seed=nothing,
                  on_error::Symbol=:record)
    seed === nothing && run_seed === nothing &&
        throw(ArgumentError("give `seed`: runs are reproducible only with a stated seed"))
    seed === nothing || run_seed === nothing ||
        throw(ArgumentError("give `seed` or `run_seed`, not both: with `run_seed` the " *
                            "seed of every run comes from the function"))
    on_error in (:record, :throw) ||
        throw(ArgumentError("on_error must be :record or :throw, got :$on_error"))
    replicates >= 1 || throw(ArgumentError("replicates must be at least 1"))
    belief = draws isa SampleBelief ? draws : SampleBelief(draws)
    named = _named_actions(actions)
    isempty(named) && throw(ArgumentError("no action was given"))
    labels = first.(named)
    allunique(labels) || throw(ArgumentError("the action labels must be distinct"))
    rows = isempty(belief.draws) ? fill((;), length(belief)) : Tables.rowtable(belief.draws)
    nd = length(rows)
    results = Array{Any}(undef, length(named), nd, replicates)
    state = fill(_OK, length(named), nd, replicates)
    names = nothing
    first_error = nothing
    seeds = [_seed_of_run(seed, run_seed, rows[d], d, r) for d in 1:nd, r in 1:replicates]
    run_seed === nothing || allunique(seeds) ||
        throw(ArgumentError("`run_seed` gave the same seed to two runs, whose outcomes " *
                            "would be copies of each other and not replicates"))
    for d in 1:nd, r in 1:replicates
        for (a, (_, act)) in enumerate(named)
            try
                rng = Xoshiro(seeds[d, r])
                acted = apply_action(with_parameters(prob, rows[d]), act)
                o = _outcome_tuple(outcome(acted, simulate(acted; rng)))
                names === nothing && (names = keys(o))
                keys(o) == names ||
                    throw(InvalidOutcomeError("`outcome` returned the names $(keys(o)) " *
                                              "in one run and $names in another"))
                results[a, d, r] = o
            catch err
                (err isa InterruptException || err isa InvalidOutcomeError ||
                 on_error === :throw) && rethrow()
                first_error === nothing && (first_error = err)
                state[a, d, r] = "error:" * string(nameof(typeof(err)))
            end
        end
    end
    names === nothing &&
        throw(InvalidOutcomeError("every run failed; the first error was: " *
                                  sprint(showerror, first_error)))
    T = Union{}
    raw = Array{Any}(undef, length(named), nd, replicates, length(names))
    fill!(raw, 0)
    for idx in CartesianIndices(state)
        state[idx] == _OK || continue
        for (k, v) in enumerate(values(results[idx]))
            (v isa Real && isfinite(v)) ||
                throw(InvalidOutcomeError("outcome $(names[k]) of action " *
                                          "$(labels[idx[1]]), draw " *
                                          "$(belief.states[idx[2]]) is $(repr(v)), not " *
                                          "a finite number"))
            raw[idx, k] = v
            T = promote_type(T, typeof(v))
        end
    end
    T === Bool && (T = Int)
    return OutcomeTable(belief.states, labels, collect(Symbol, names),
                        _default_labels("r", replicates), Array{T,4}(raw), state,
                        belief.draws, belief.weights)
end
