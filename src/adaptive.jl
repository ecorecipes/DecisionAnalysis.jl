# Sequential (adaptive) management: the same decision taken period after period, with
# what each action reveals used in the next. The recursion is the one of
# `proofs/DecisionAnalysisProofs/Finite/Adaptive.lean` (a fixed state) and
# `Finite/AdaptiveDynamic.lean` (a state that moves), on unnormalised weights (the
# joint form of `voi.jl`): nothing is divided by the probability of a signal.

# The experiment that goes with each feasible action; `nothing` where the action
# reveals nothing.
function _action_experiments(problem::DecisionProblem, experiment::Experiment)
    _check_states(experiment, problem.payoff.states, "payoff table")
    return Union{Nothing,Experiment}[experiment for _ in feasible_actions(problem)]
end
function _action_experiments(problem::DecisionProblem, experiments)
    named = Dict{Symbol,Any}(Symbol(k) => v
                             for (k, v) in (experiments isa AbstractVector{<:Pair} ?
                                            experiments : pairs(experiments)))
    for (a, e) in named
        a in problem.payoff.actions ||
            throw(ArgumentError("experiments names $a, which is not an action of the " *
                                "payoff table"))
        e isa Experiment ||
            throw(ArgumentError("the entry of experiments for $a is not an Experiment"))
        _check_states(e, problem.payoff.states, "payoff table")
    end
    return Union{Nothing,Experiment}[get(named, a, nothing)
                                     for a in feasible_actions(problem)]
end

# The transition matrix that goes with each feasible action; `nothing` everywhere when
# the state does not move.
function _checked_transition(problem::DecisionProblem, T, who::AbstractString)
    T isa AbstractMatrix{<:Real} ||
        throw(ArgumentError("$who must be a matrix of probabilities, got $(typeof(T))"))
    n = length(problem.payoff.states)
    size(T) == (n, n) ||
        throw(ArgumentError("$who must be $n × $n (next state × present state), " *
                            "got $(join(size(T), " × "))"))
    defect = _stochastic_defect(T, _default_atol(eltype(T)))
    defect === nothing ||
        throw(ArgumentError("$who must have rows = next state and columns = present " *
                            "state, each column a probability vector, but $defect"))
    return Matrix(T)
end
function _action_transitions(problem::DecisionProblem, ::Nothing)
    return [nothing
            for _ in
                feasible_actions(problem)]
end
function _action_transitions(problem::DecisionProblem, T::AbstractMatrix)
    checked = _checked_transition(problem, T, "transitions")
    return [checked for _ in feasible_actions(problem)]
end
function _action_transitions(problem::DecisionProblem, transitions)
    named = Dict{Symbol,Any}(Symbol(k) => v
                             for (k, v) in (transitions isa AbstractVector{<:Pair} ?
                                            transitions : pairs(transitions)))
    for a in keys(named)
        a in problem.payoff.actions ||
            throw(ArgumentError("transitions names $a, which is not an action of the " *
                                "payoff table"))
    end
    return map(feasible_actions(problem)) do a
        haskey(named, a) ||
            throw(ArgumentError("transitions has no entry for the action $a; give " *
                                "one matrix for all actions or one for each"))
        return _checked_transition(problem, named[a], "the transition matrix of $a")
    end
end

_weights_key(w::AbstractVector{<:Union{Integer,Rational}}) = w
_weights_key(w::AbstractVector) = round.(w; sigdigits=12)

function _horizon_weight(discount, horizon::Integer)
    return sum(discount^k for k in 0:(horizon - 1))
end

# The model: for each feasible action, its payoffs, what it shows (or `nothing`) and
# how it moves the state (or `nothing`), with the discount and the state that a signal
# is about. Everything below takes it as an argument; nothing is a closure, so no
# function can write to another's variables.
struct _Sequential{T}
    Q::Matrix{T}
    likelihoods::Vector{Union{Nothing,Matrix{T}}}
    moves::Vector{Union{Nothing,Matrix{T}}}
    discount::T
    signal_next::Bool
end

# The weights moved to the next period under action `a`, no signal used.
function _push_forward(m::_Sequential, w::AbstractVector, a::Integer)
    M = m.moves[a]
    return M === nothing ? w : M * w
end

# The weights of the next period after each signal of positive probability under
# action `a`, as `(signal index, weights)`; the index is 0 where nothing is shown.
function _updates(m::_Sequential{T}, w::Vector{T}, a::Integer) where {T}
    L = m.likelihoods[a]
    L === nothing && return [(0, _push_forward(m, w, a))]
    out = Tuple{Int,Vector{T}}[]
    before = m.signal_next ? _push_forward(m, w, a) : w
    for z in axes(L, 1)
        seen = before .* view(L, z, :)
        next = m.signal_next ? seen : _push_forward(m, seen, a)
        iszero(sum(next)) || push!(out, (z, next))
    end
    return out
end

# The value of the best policy
# ----------------------------
#
# Two ways of holding the value of `n` periods from the weights `w`: a table of the
# beliefs met so far (`method=:beliefs`), or one set of vectors for each `n`
# (`:vectors`, `:points`). `_best` is the same look-ahead for both.

struct _ByBeliefs{T}
    memo::Dict{Tuple{Int,Vector{T}},Tuple{T,Int}}
end
struct _BySets{T}
    sets::Vector{Tuple{Matrix{T},Vector{Int}}}
end

_later(c::_ByBeliefs, m::_Sequential, n::Integer, w) = first(_best(c, m, n, w))
function _later(c::_BySets{T}, ::_Sequential, n::Integer, w) where {T}
    return n == 0 ? zero(T) : maximum(first(c.sets[n]) * w)
end

# The best first action for `n` periods from `w` and its value: this period's
# expected payoff plus the discounted value of what follows each signal. Ties go to
# the first action.
function _look_ahead(c, m::_Sequential{T}, n::Integer, w::Vector{T}) where {T}
    scores = m.Q * w
    value, choice = zero(T), 0
    for a in axes(m.Q, 1)
        v = scores[a]
        if n > 1
            for (_, next) in _updates(m, w, a)
                v += m.discount * _later(c, m, n - 1, next)
            end
        end
        (choice == 0 || v > value) && ((value, choice) = (v, a))
    end
    return value, choice
end

function _best(c::_ByBeliefs{T}, m::_Sequential{T}, n::Integer, w::Vector{T}) where {T}
    n == 0 && return (zero(T), 0)
    key = (Int(n), _weights_key(w))
    found = get(c.memo, key, nothing)
    found === nothing || return found
    result = _look_ahead(c, m, n, w)
    c.memo[key] = result
    return result
end
function _best(c::_BySets{T}, m::_Sequential{T}, n::Integer, w::Vector{T}) where {T}
    return n == 0 ? (zero(T), 0) : _look_ahead(c, m, n, w)
end

# The fixed plan
# --------------
#
# The best plan fixed in advance for `n` periods from `w`, and its first action:
# `_plan(p, m, n, w)`, with one method for each way of finding it.

# The state stays put: the best action for `w`, repeated.
struct _RepeatPlan end
# The state moves in the same way whatever is done: the best action for the weights
# of each period in turn.
struct _SharedPlan end
struct _PlanByBeliefs{T}
    memo::Dict{Tuple{Int,Vector{T}},Tuple{T,Int}}
end
struct _PlanBySets{T}
    sets::Vector{Tuple{Matrix{T},Vector{Int}}}
end

function _plan(::_RepeatPlan, m::_Sequential{T}, n::Integer, w::Vector{T}) where {T}
    n == 0 && return (zero(T), 0)
    scores = m.Q * w
    a = argmax(scores)
    return (_horizon_weight(m.discount, n) * scores[a], a)
end
function _plan(::_SharedPlan, m::_Sequential{T}, n::Integer, w::Vector{T}) where {T}
    n == 0 && return (zero(T), 0)
    scores = m.Q * w
    total, moved = maximum(scores), w
    for t in 1:(n - 1)
        moved = _push_forward(m, moved, 1)
        total += m.discount^t * maximum(m.Q * moved)
    end
    return (total, argmax(scores))
end
_plan_later(p::_PlanByBeliefs, m, n, w) = first(_plan(p, m, n, w))
function _plan_later(p::_PlanBySets{T}, ::_Sequential, n::Integer, w) where {T}
    return n == 0 ? zero(T) : maximum(first(p.sets[n]) * w)
end
function _plan_look_ahead(p, m::_Sequential{T}, n::Integer, w::Vector{T}) where {T}
    scores = m.Q * w
    value, choice = zero(T), 0
    for a in axes(m.Q, 1)
        v = scores[a]
        n > 1 && (v += m.discount * _plan_later(p, m, n - 1, _push_forward(m, w, a)))
        (choice == 0 || v > value) && ((value, choice) = (v, a))
    end
    return value, choice
end
function _plan(p::_PlanByBeliefs{T}, m::_Sequential{T}, n::Integer,
               w::Vector{T}) where {T}
    n == 0 && return (zero(T), 0)
    key = (Int(n), _weights_key(w))
    found = get(p.memo, key, nothing)
    found === nothing || return found
    result = _plan_look_ahead(p, m, n, w)
    p.memo[key] = result
    return result
end
function _plan(p::_PlanBySets{T}, m::_Sequential{T}, n::Integer, w::Vector{T}) where {T}
    return n == 0 ? (zero(T), 0) : _plan_look_ahead(p, m, n, w)
end

# The passive manager: in each period the first action of the fixed plan for the
# weights of that period, updated on what is seen. Its value is not a maximum of
# linear functions, so it is always found by following its beliefs.
function _replanned!(memo::Dict{Tuple{Int,Vector{T}},T}, p, m::_Sequential{T},
                     n::Integer, w::Vector{T}) where {T}
    n == 0 && return zero(T)
    key = (Int(n), _weights_key(w))
    found = get(memo, key, nothing)
    found === nothing || return found
    a = last(_plan(p, m, n, w))
    v = (m.Q * w)[a]
    if n > 1
        for (_, next) in _updates(m, w, a)
            v += m.discount * _replanned!(memo, p, m, n - 1, next)
        end
    end
    memo[key] = v
    return v
end

# The state seen before every action: backward induction on the states.
function _seen_value(m::_Sequential{T}, horizon::Integer, w0::Vector{T}) where {T}
    states = eachindex(w0)
    value = zeros(T, length(w0))
    for _ in 1:horizon
        previous = value
        value = [maximum(m.Q[a, t] +
                         m.discount * (m.moves[a] === nothing ? previous[t] :
                                       sum(m.moves[a][s, t] * previous[s] for s in states))
                         for a in axes(m.Q, 1)) for t in states]
    end
    return sum(w0 .* value)
end

# One action in every period.
function _repeated_value(m::_Sequential{T}, a::Integer, horizon::Integer,
                         w0::Vector{T}) where {T}
    total, w = zero(T), w0
    for t in 1:horizon
        total += m.discount^(t - 1) * sum(view(m.Q, a, :) .* w)
        w = _push_forward(m, w, a)
    end
    return total
end

# The active policy as a tree to `levels` periods. `signals[a]` labels what action `a`
# shows.
function _policy_node(c, m::_Sequential{T}, signals, actions::Vector{Symbol},
                      horizon::Integer, n::Integer, w::Vector{T},
                      levels::Integer) where {T}
    value, a = _best(c, m, n, w)
    total = sum(w)
    branches = Pair{Symbol,Any}[]
    if levels > 1 && n > 1 && m.likelihoods[a] !== nothing
        for (z, next) in _updates(m, w, a)
            push!(branches,
                  signals[a][z] => _policy_node(c, m, signals, actions, horizon, n - 1,
                                                next, levels - 1))
        end
    end
    return (; period=horizon - n + 1, probability=total, belief=w ./ total,
            action=actions[a], value=value / total, branches)
end

# Value functions as sets of vectors
# ----------------------------------
#
# A set is a matrix whose rows are vectors over the states; its value at the weights
# `w` is `maximum(G * w)`. The step and the pruning are those of
# `proofs/DecisionAnalysisProofs/Finite/ValueVectors.lean`. `method=:vectors` keeps
# every vector that matters and is for two states only: there they are an upper
# envelope of lines, found exactly by sorting. With more states the pruning needs a
# linear programme for each vector and, in exact arithmetic, was slower than the
# recursion over beliefs on every problem tried. `method=:points` keeps only the
# vectors that are best at the points of a grid, for any number of states.

# The matrices `K` with next weights `K * w`, one for each signal of each action.
function _kernels(m::_Sequential{T}) where {T}
    n = size(m.Q, 2)
    same = Matrix{T}(I, n, n)
    return map(axes(m.Q, 1)) do a
        move = m.moves[a] === nothing ? same : m.moves[a]
        L = m.likelihoods[a]
        L === nothing && return [move]
        return [m.signal_next ? Diagonal(L[z, :]) * move : move * Diagonal(L[z, :])
                for z in axes(L, 1)]
    end
end

# The indices of the rows of `G` that are needed for the maximum at some nonnegative
# weights over two states: the rows on the upper envelope of the lines
# `G[i, 1] + (G[i, 2] − G[i, 1]) x` over `0 ≤ x ≤ 1`, found as a convex hull. A row
# that reaches the envelope at a single point is not needed. Among equal rows the
# first is kept.
function _envelope(G::AbstractMatrix{T}) where {T}
    m = size(G, 1)
    m <= 1 && return collect(1:m)
    slope(i) = G[i, 2] - G[i, 1]
    crossing(i, j) = (G[i, 1] - G[j, 1]) / (slope(j) - slope(i))
    order = sort(1:m; by=i -> (slope(i), -G[i, 1], i))
    hull = Int[]
    for i in order
        # Of lines with the same slope only the highest, which comes first, is needed.
        isempty(hull) || slope(hull[end]) != slope(i) || continue
        while length(hull) >= 2 &&
            crossing(hull[end - 1], i) <= crossing(hull[end - 1], hull[end])
            pop!(hull)
        end
        push!(hull, i)
    end
    # Line k of the hull is the highest between its crossings with its neighbours;
    # it is needed if that stretch has positive length inside [0, 1].
    keep = Int[]
    for k in eachindex(hull)
        from = k == 1 ? zero(T) : max(zero(T), crossing(hull[k - 1], hull[k]))
        to = k == length(hull) ? one(T) : min(one(T), crossing(hull[k], hull[k + 1]))
        from < to && push!(keep, hull[k])
    end
    return sort!(keep)
end

# One step back, exactly: the vectors for one more period from the set `G` for the
# periods that follow, each tagged with its action.
function _backup_vectors(G::AbstractMatrix{T}, Q::AbstractMatrix{T}, kernels,
                         discount::T) where {T}
    blocks, tags = Matrix{T}[], Int[]
    for a in axes(Q, 1)
        sums = Matrix{T}(reshape(Q[a, :], 1, :))
        for K in kernels[a]
            later = discount .* (G * K)
            later = later[_envelope(later), :]
            crossed = Matrix{T}(undef, size(sums, 1) * size(later, 1), size(G, 2))
            for i in axes(sums, 1), j in axes(later, 1)
                crossed[(i - 1) * size(later, 1) + j, :] = view(sums, i, :) .+
                                                           view(later, j, :)
            end
            sums = crossed[_envelope(crossed), :]
        end
        push!(blocks, sums)
        append!(tags, fill(a, size(sums, 1)))
    end
    all_rows = reduce(vcat, blocks)
    keep = _envelope(all_rows)
    return all_rows[keep, :], tags[keep]
end

# The beliefs over `n` states whose entries are multiples of `1 / k`, the corners of
# the simplex first.
function _simplex_grid(::Type{T}, n::Integer, k::Integer) where {T}
    grid = [T[s == t for s in 1:n] for t in 1:n]
    _fill_grid!(grid, zeros(Int, n), 1, k, k, T)
    return grid
end
function _fill_grid!(grid::Vector{Vector{T}}, counts::Vector{Int}, position::Integer,
                     left::Integer, k::Integer, ::Type{T}) where {T}
    if position == length(counts)
        counts[position] = left
        maximum(counts) == k || push!(grid, T[T(c) / T(k) for c in counts])
        return grid
    end
    for c in 0:left
        counts[position] = c
        _fill_grid!(grid, counts, position + 1, left - c, k, T)
    end
    return grid
end

# One step back at chosen points: for each point, the vector of the best action there,
# each signal followed by the best vector of `G` at the weights it leads to. The
# result is a lower bound on the value function, tight at the points if `G` is.
function _point_backup(G::AbstractMatrix{T}, Q::AbstractMatrix{T}, kernels, discount::T,
                       points::Vector{Vector{T}}) where {T}
    products = [[G * K for K in ks] for ks in kernels]
    rows, tags, seen = Vector{T}[], Int[], Set{Vector{T}}()
    for q in points
        best_vector, best_value, best_action = T[], zero(T), 0
        for a in axes(Q, 1)
            candidate = Q[a, :]
            for GK in products[a]
                candidate .+= discount .* view(GK, argmax(GK * q), :)
            end
            value = dot(candidate, q)
            if best_action == 0 || value > best_value
                best_vector, best_value, best_action = candidate, value, a
            end
        end
        if !(best_vector in seen)
            push!(seen, best_vector)
            push!(rows, best_vector)
            push!(tags, best_action)
        end
    end
    return permutedims(reduce(hcat, rows)), tags
end

# An upper bound on the value function from the same points: for each number of
# periods left, a bound at each point (`values`) and how far it lies below the plane
# through the corner values (`gaps`, never positive for a convex function).
struct _UpperBound{T}
    points::Vector{Vector{T}}
    values::Vector{Vector{T}}
    gaps::Vector{Vector{T}}
end

# The bound for `n` periods from the weights `w`. The value function is convex and
# scales with the weights, so it lies below the plane through the corner values and
# below each "sawtooth" that hangs from one more point.
function _upper_at(u::_UpperBound{T}, n::Integer, w::Vector{T}) where {T}
    n == 0 && return zero(T)
    total = sum(w)
    iszero(total) && return zero(T)
    b = w ./ total
    values, gaps = u.values[n], u.gaps[n]
    states = eachindex(b)
    plane = sum(b[t] * values[t] for t in states)
    bound = plane
    for i in (length(b) + 1):length(u.points)
        q = u.points[i]
        share = minimum(b[t] / q[t] for t in states if !iszero(q[t]))
        hanging = plane + share * gaps[i]
        hanging < bound && (bound = hanging)
    end
    return total * bound
end

# The upper bound on the value of `horizon` periods from `w0`. What each action leads
# to from each point does not depend on the number of periods left, so it is found
# once.
function _upper_value(m::_Sequential{T}, points::Vector{Vector{T}}, horizon::Integer,
                      w0::Vector{T}) where {T}
    u = _UpperBound{T}(points, Vector{T}[], Vector{T}[])
    actions = axes(m.Q, 1)
    successors = [[_updates(m, q, a) for a in actions] for q in points]
    function step(n, q, after)
        return maximum(dot(view(m.Q, a, :), q) +
                       (n == 1 ? zero(T) :
                        m.discount * sum(_upper_at(u, n - 1, next)
                                         for (_, next) in after[a]; init=zero(T)))
                       for a in actions)
    end
    corners = 1:length(w0)
    for n in 1:(horizon - 1)
        values = [step(n, q, successors[i]) for (i, q) in enumerate(points)]
        push!(u.values, values)
        push!(u.gaps,
              [values[i] - sum(q[t] * values[t] for t in corners)
               for (i, q) in enumerate(points)])
    end
    return step(horizon, w0, [_updates(m, w0, a) for a in actions])
end

"""
    adaptive_management(problem::DecisionProblem, experiments;
                        horizon, discount=1, depth=min(horizon, 3),
                        transitions=nothing, signal_of=:current,
                        method=:beliefs, passive=(method !== :points),
                        grid=10) -> NamedTuple

**Sequential (adaptive) management**: the decision of `problem` is taken in each of
`horizon` periods. The state is unknown, and fixed unless `transitions` is given; in
each period the action `a` pays `Q[a, θ]` and reveals a signal, the belief is updated,
and the next action is chosen on the new belief. What is learned depends on what is
done:

- `experiments` is one [`Experiment`](@ref), observed whatever the action, or a
  collection of `action => Experiment` pairs (a vector of pairs, a named tuple or a
  dictionary). An action that is not named reveals nothing.
- `discount` multiplies the payoffs of each later period (`δ ≥ 0`; period `t` has
  weight `δ^(t−1)`).
- `transitions` makes the state move between periods. It is one matrix, used whatever
  the action, or a collection of `action => matrix` pairs naming every feasible action.
  A matrix has `T[θ′, θ] = P(next state θ′ | present state θ)`: rows are the next
  state and columns the present one, so every **column** sums to one, as in an
  [`Experiment`](@ref). The prior is the belief about the state of the first period.
- `signal_of` says which state the signal of a period is about: `:current` (default),
  the state that the period's payoff came from, which then moves; or `:next`, the
  state after the move (the usual convention of partially observed Markov decision
  processes). The two coincide when the state is fixed.

- `method` is how the best policy is found; both are exact. `:beliefs` (default)
  evaluates every belief that can be reached, whose number grows with every period.
  `:vectors` keeps the value function as the maximum of a set of linear functions of
  the belief, one set for each number of periods left, and discards the functions
  that are never the maximum. The sets are usually far smaller than the set of
  beliefs, so long horizons become practical. It is available for **two states**
  only, where the functions are lines and the ones that matter are found by sorting.
- `method=:points` is for **any number of states** over long horizons, and is **not
  exact**: it returns two numbers that the value of the best policy lies between.
  `active` is a lower bound, the value of a policy that can be carried out: the value
  function is backed up only at the beliefs of a grid (entries that are multiples of
  `1 / grid`, and the prior). `active_upper` is an upper bound, from the same grid,
  using that the value function is convex. Both are computed exactly for exact inputs;
  only the grid is an approximation, and the gap between the two says how good it is.
  A finer `grid` narrows it and costs more: the number of points is
  `binomial(grid + n − 1, n − 1)` for `n` states. With this method `perfect` and
  `baseline` are exact. The fixed plan (`no_learning`, `plan`, `act_now_action`) is
  found, exactly, only where the state is fixed or moves in the same way whatever is
  done; otherwise those fields, `value_of_learning` and `evpi` are `nothing`.
- `passive=false` leaves out the passive manager. Its value is not a maximum of linear
  functions, so with every method it is found by following its beliefs, and where
  those multiply it can take longer than everything else. It is left out by default
  with `method=:points`, where asking for it throws an `ArgumentError` unless the
  fixed plan can be found (see above).

Four ways of managing are valued, each as a total over the horizon:

- `no_learning`: the best plan fixed in advance, which ignores what is seen. With a
  fixed state that is the best single action on the prior, repeated; with
  `transitions` the plan allows for how the state is expected to move, and may change
  action from period to period;
- `passive`: in each period the action that would be best if nothing more were going
  to be learned, with the belief updated as results arrive, but no action ever taken
  *in order to* learn (passive adaptive management). With a fixed state that is the
  best action for the current belief; with `transitions` it is the first action of
  the best fixed plan for the remaining periods, made again after each signal;
- `active`: the best policy, which may take an action that pays less now because of
  what it reveals (active adaptive management). It is found by the recursion
  `V_n(w) = max_a [Σ_θ w(θ) Q[a, θ] + δ Σ_z V_{n−1}(w′_{a,z})]`, with
  `w′_{a,z} = w · L_a[z, ·]` for a fixed state, `T_a (w · L_a[z, ·])` with
  `transitions`, and `(T_a w) · L_a[z, ·]` with `signal_of=:next`;
- `perfect`: the state known. With `transitions`, the state is seen before every
  action (the fully observed Markov decision process), not only at the start.

Always `no_learning ≤ passive ≤ active ≤ perfect` (proved in the Lean development,
`Finite/Adaptive.lean` and, for a state that moves, `Finite/AdaptiveDynamic.lean`).
With a moving state a manager who took only the best action for the current period
could do worse than the fixed plan, which is why `passive` plans ahead. The result is
`(; horizon, discount, no_learning, passive, active, active_upper, perfect,
value_of_learning, value_of_probing, evpi, baseline, act_now_action, first_action, plan,
policy, beliefs_evaluated, value_function)`: `value_of_learning = active − no_learning`;
`value_of_probing = active − passive`, what deliberate experimentation adds;
`evpi = perfect − no_learning`, which bounds both; `baseline` is the value of taking
the baseline action in every period (`nothing` without a baseline); `act_now_action`
is the first action of the passive and no-learning managers and `first_action` that of
the active one, so the two differ exactly when it pays to probe at once. `plan` is the
fixed plan of the no-learning manager, one action for each period. With
`passive=false`, `passive` and `value_of_probing` are `nothing`.

`active_upper` equals `active` with the exact methods.

`value_function` is `nothing` with `method=:beliefs`. With `:vectors` (and with
`:points`, where it gives the lower bound) its entry `n` is
`(; actions, vectors)` for `n` periods left: the value at the belief `b` is
`maximum(vectors * b)`, and the action of the row that attains it is a best action
there. So the best action at any belief and any time can be read off without
running the recursion again. (`first_action` and the policy tree break ties towards
the first action; the rows of `value_function` need not.)

`policy` is the active policy as a tree, to `depth` periods: each node is
`(; period, probability, belief, action, value, branches)`, with the probability of
reaching it, the belief there, the action taken, the expected value from there on
given that the node is reached, and `branches`, a vector of `signal => node` for the
signals of positive probability. `branches` is empty at the last period shown, and
after an action that shows nothing, where the tree stops. `policy` is `nothing` with
`depth=0`.

Ties go to the first action. Exact in, exact out. With `method=:beliefs` the recursion visits every distinct
belief once (`beliefs_evaluated`, which is `nothing` with the other methods); beliefs reached by the same observations in a
different order coincide, so repeated trials of one kind stay cheap, but the number
grows quickly with the number of signals, actions and periods. With floating-point
inputs, beliefs that agree to twelve significant digits are treated as one. The exact
weights are products of one likelihood per period, so over many periods rational
inputs should be `Rational{BigInt}` (machine rationals overflow with an error). Needs a
single prior; infeasible actions are left out. Payoffs, experiments and transitions are
the same in every period.

```jldoctest
julia> t = PayoffTable([:safe, :risky], [:bad, :good], [0 0; -3//2 1]);

julia> p = DecisionProblem(t, [1 // 2, 1 // 2]; baseline=:safe);

julia> shows = Experiment([1 0; 0 1]; signals=[:failed, :worked]);

julia> r = adaptive_management(p, (; risky=shows); horizon=3);

julia> r.no_learning, r.passive, r.active, r.perfect
(0//1, 0//1, 3//4, 3//2)

julia> r.act_now_action, r.first_action, r.value_of_probing
(:safe, :risky, 3//4)

julia> [signal => node.action for (signal, node) in r.policy.branches]
2-element Vector{Pair{Symbol, Symbol}}:
 :failed => :safe
 :worked => :risky
```
"""
function adaptive_management(problem::DecisionProblem, experiments; horizon::Integer,
                             discount::Real=1, depth::Integer=min(horizon, 3),
                             transitions=nothing, signal_of::Symbol=:current,
                             method::Symbol=:beliefs,
                             passive::Bool=method !== :points, grid::Integer=10)
    horizon >= 1 || throw(ArgumentError("the horizon must be at least 1, got $horizon"))
    discount >= 0 ||
        throw(ArgumentError("the discount cannot be negative, got $discount"))
    depth >= 0 || throw(ArgumentError("the depth cannot be negative, got $depth"))
    signal_of in (:current, :next) ||
        throw(ArgumentError("signal_of must be :current or :next, got :$signal_of"))
    method in (:beliefs, :vectors, :points) ||
        throw(ArgumentError("method must be :beliefs, :vectors or :points, got :$method"))
    p = _prior(problem, :adaptive_management)
    method === :vectors && length(problem.payoff.states) != 2 &&
        throw(ArgumentError("method = :vectors needs exactly two states, got " *
                            "$(length(problem.payoff.states)); use method = :beliefs"))
    method === :points && grid < 1 &&
        throw(ArgumentError("the grid must have at least one division, got $grid"))
    actions = feasible_actions(problem)
    studies = _action_experiments(problem, experiments)
    transition = _action_transitions(problem, transitions)
    moving = transitions !== nothing
    # With method = :points the fixed plan, and so the passive manager, is found only
    # where the state stays put or moves in the same way whatever is done.
    plan_known = method !== :points || !moving || all(==(first(transition)), transition)
    passive && !plan_known &&
        throw(ArgumentError("with method = :points the passive manager is found only " *
                            "where the state stays put or moves in the same way " *
                            "whatever is done; use passive = false"))
    Q = _feasible_Q(problem)
    given = promote_type(eltype(Q), eltype(p), typeof(discount),
                         (eltype(e.likelihood) for e in studies if e !== nothing)...,
                         (eltype(M) for M in transition if M !== nothing)...)
    # The points of a grid are fractions, so integers are carried as rationals.
    T = method === :points && given <: Integer ? Rational{given} : given
    model = _Sequential{T}(Q,
                           [e === nothing ? nothing : Matrix{T}(e.likelihood)
                            for e in studies],
                           [M === nothing ? nothing : Matrix{T}(M) for M in transition],
                           T(discount), signal_of === :next)
    signals = [e === nothing ? nothing : e.signals for e in studies]
    return _manage(model, collect(T, p), actions, signals, problem.baseline, horizon, depth,
                   method, passive, plan_known, moving, grid, discount)
end

# Everything after the checks, with the element type fixed.
function _manage(m::_Sequential{T}, w0::Vector{T}, actions::Vector{Symbol}, signals,
                 baseline, horizon::Integer, depth::Integer, method::Symbol,
                 passive::Bool, plan_known::Bool, moving::Bool, grid::Integer,
                 discount) where {T}
    states = length(w0)
    # The value of the best policy, and of the best fixed plan.
    sets = Tuple{Matrix{T},Vector{Int}}[]
    plan_sets = Tuple{Matrix{T},Vector{Int}}[]
    active_upper = nothing
    if method === :beliefs
        policy = _ByBeliefs{T}(Dict{Tuple{Int,Vector{T}},Tuple{T,Int}}())
        plan = moving ? _PlanByBeliefs{T}(Dict{Tuple{Int,Vector{T}},Tuple{T,Int}}()) :
               _RepeatPlan()
    else
        kernels = _kernels(m)
        G = zeros(T, 1, states)
        if method === :vectors
            # The fixed plan sees no signal: one kernel for each action, its move.
            H = zeros(T, 1, states)
            moves_only = [[M] for M in m.moves if M !== nothing]
            for _ in 1:horizon
                G, tags = _backup_vectors(G, m.Q, kernels, m.discount)
                push!(sets, (G, tags))
                if moving
                    H, plan_tags = _backup_vectors(H, m.Q, moves_only, m.discount)
                    push!(plan_sets, (H, plan_tags))
                end
            end
            plan = moving ? _PlanBySets{T}(plan_sets) : _RepeatPlan()
        else
            # A lower bound, from vectors kept at the points of a grid, each the value
            # of a plan that can be carried out; and an upper bound from the same points.
            points = _simplex_grid(T, states, grid)
            start = w0 ./ sum(w0)
            start in points || push!(points, start)
            for _ in 1:horizon
                G, tags = _point_backup(G, m.Q, kernels, m.discount, points)
                push!(sets, (G, tags))
            end
            active_upper = _upper_value(m, points, horizon, w0)
            plan = !plan_known ? nothing : moving ? _SharedPlan() : _RepeatPlan()
        end
        policy = _BySets{T}(sets)
    end

    active_value, first_choice = _best(policy, m, horizon, w0)
    active_upper === nothing && (active_upper = active_value)
    perfect = _seen_value(m, horizon, w0)
    base = baseline === nothing ? nothing : findfirst(==(baseline), actions)
    baseline_value = base === nothing ? nothing : _repeated_value(m, base, horizon, w0)

    no_learning, act_now, fixed_plan, passive_value = nothing, nothing, nothing, nothing
    if plan !== nothing
        no_learning, a0 = _plan(plan, m, horizon, w0)
        act_now = actions[a0]
        fixed_plan = Symbol[]
        w = w0
        for n in horizon:-1:1
            a = last(_plan(plan, m, n, w))
            push!(fixed_plan, actions[a])
            w = _push_forward(m, w, a)
        end
        passive &&
            (passive_value = _replanned!(Dict{Tuple{Int,Vector{T}},T}(), plan, m, horizon,
                                         w0))
    end
    tree = depth == 0 ? nothing :
           _policy_node(policy, m, signals, actions, horizon, horizon, w0, depth)
    return (; horizon, discount, no_learning, passive=passive_value, active=active_value,
            active_upper, perfect,
            value_of_learning=no_learning === nothing ? nothing :
                              active_value - no_learning,
            value_of_probing=passive_value === nothing ? nothing :
                             active_value - passive_value,
            evpi=no_learning === nothing ? nothing : perfect - no_learning,
            baseline=baseline_value, act_now_action=act_now,
            first_action=actions[first_choice], plan=fixed_plan, policy=tree,
            beliefs_evaluated=method === :beliefs ? length(policy.memo) : nothing,
            value_function=method === :beliefs ? nothing :
                           [(; actions=actions[tags], vectors=G) for (G, tags) in sets])
end
