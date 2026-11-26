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

# Value functions as sets of vectors (`method=:vectors`). A set is a matrix whose rows
# are vectors over the states; its value at the weights `w` is `maximum(G * w)`. The
# step and the pruning are those of `proofs/DecisionAnalysisProofs/Finite/ValueVectors.lean`.
# Only two states are supported: there the vectors that matter are an upper envelope of
# lines, found exactly by sorting. With more states the pruning needs a linear programme
# for each vector and, in exact arithmetic, was slower than the recursion over beliefs
# on every problem tried.

# The rows of `G` that are needed for the maximum at some nonnegative weights over two
# states, with their tags: the rows on the upper envelope of the lines
# `G[i, 1] + (G[i, 2] − G[i, 1]) x` over `0 ≤ x ≤ 1`, found as a convex hull. A row
# that reaches the envelope at a single point is not needed. Among equal rows the
# first is kept.
function _prune_vectors(G::AbstractMatrix{T}, tags::AbstractVector{Int}) where {T}
    m = size(G, 1)
    m <= 1 && return Matrix(G), collect(tags)
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
    sort!(keep)
    return G[keep, :], tags[keep]
end

# The beliefs over `n` states whose entries are multiples of `1 / k`, the corners of
# the simplex first.
function _simplex_grid(::Type{T}, n::Integer, k::Integer) where {T}
    corners = [T[s == t for s in 1:n] for t in 1:n]
    others = Vector{T}[]
    counts = zeros(Int, n)
    function fill!(position, left)
        if position == n
            counts[n] = left
            maximum(counts) == k || push!(others, T[T(c) / T(k) for c in counts])
            return nothing
        end
        for c in 0:left
            counts[position] = c
            fill!(position + 1, left - c)
        end
        return nothing
    end
    fill!(1, k)
    return vcat(corners, others)
end

# One step back at chosen points (`method=:points`): for each point, the vector of the
# best action there, each signal followed by the best vector of `G` at the weights it
# leads to. The result is a lower bound on the value function, tight at the points if
# `G` is.
function _point_backup(G::AbstractMatrix{T}, Q::AbstractMatrix, kernels, discount,
                       points) where {T}
    rows, tags = Vector{T}[], Int[]
    for q in points
        best_vector, best_value, best_action = T[], zero(T), 0
        for a in axes(Q, 1)
            candidate = collect(T, view(Q, a, :))
            for K in kernels[a]
                later = G * (K * q)
                candidate .+= discount .* vec(transpose(view(G, argmax(later), :)) * K)
            end
            value = sum(candidate .* q)
            if best_action == 0 || value > best_value
                best_vector, best_value, best_action = candidate, value, a
            end
        end
        if !(best_vector in rows)
            push!(rows, best_vector)
            push!(tags, best_action)
        end
    end
    return permutedims(reduce(hcat, rows)), tags
end

# One step back: the vectors for one more period from the set `G` for the periods that
# follow. `kernels[a]` holds, for each signal of action `a`, the matrix `K` with
# next weights `K * w`. Each vector is tagged with its action.
function _backup_vectors(G::AbstractMatrix{T}, Q::AbstractMatrix, kernels,
                         discount) where {T}
    blocks, tags = Matrix{T}[], Int[]
    for a in axes(Q, 1)
        sums = Matrix{T}(reshape(collect(T, view(Q, a, :)), 1, :))
        for K in kernels[a]
            later, _ = _prune_vectors(discount .* (G * K), zeros(Int, size(G, 1)))
            crossed = Matrix{T}(undef, size(sums, 1) * size(later, 1), size(G, 2))
            for i in axes(sums, 1), j in axes(later, 1)
                crossed[(i - 1) * size(later, 1) + j, :] = view(sums, i, :) .+
                                                           view(later, j, :)
            end
            sums, _ = _prune_vectors(crossed, zeros(Int, size(crossed, 1)))
        end
        push!(blocks, sums)
        append!(tags, fill(a, size(sums, 1)))
    end
    return _prune_vectors(reduce(vcat, blocks), tags)
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
  with `method=:points`.

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
could do worse than the fixed plan, which is why `passive` plans ahead. The result is `(; horizon, discount, no_learning, passive,
active, perfect, value_of_learning, value_of_probing, evpi, baseline, act_now_action,
first_action, plan, policy, beliefs_evaluated, value_function, active_upper)`: `value_of_learning = active − no_learning`;
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
signals of positive probability (empty at the last period shown).

Ties go to the first action. Exact in, exact out. The recursion visits every distinct
belief once (`beliefs_evaluated`); beliefs reached by the same observations in a
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
    Q = _feasible_Q(problem)
    actions = feasible_actions(problem)
    studies = _action_experiments(problem, experiments)
    likelihoods = [e === nothing ? nothing : e.likelihood for e in studies]
    moves = _action_transitions(problem, transitions)
    moving = transitions !== nothing
    T = promote_type(eltype(Q), eltype(p), typeof(discount),
                     (eltype(L) for L in likelihoods if L !== nothing)...,
                     (eltype(M) for M in moves if M !== nothing)...)
    method === :points && grid < 1 &&
        throw(ArgumentError("the grid must have at least one division, got $grid"))
    # The points of a grid are fractions, so integers are carried as rationals.
    method === :points && T <: Integer && (T = Rational{T})
    w0 = collect(T, p)
    # The weights moved to the next period under action a, no signal used.
    push_forward(w, a) = moves[a] === nothing ? w : moves[a] * w
    # The weights of the next period after each signal of positive probability under
    # action a.
    function updates(w, a)
        local L = likelihoods[a]
        L === nothing && return [(0, push_forward(w, a))]
        local out = Tuple{Int,Vector{T}}[]
        local before = signal_of === :next ? push_forward(w, a) : w
        for z in axes(L, 1)
            local seen = before .* view(L, z, :)
            local next = signal_of === :next ? seen : push_forward(seen, a)
            iszero(sum(next)) || push!(out, (z, next))
        end
        return out
    end
    # With method = :vectors, the value functions for 1, 2, ... periods left: of the
    # best policy, and (when the state moves) of the best fixed plan.
    by_vectors = method !== :beliefs
    policy_vectors = Tuple{Matrix{T},Vector{Int}}[]
    plan_vectors = Tuple{Matrix{T},Vector{Int}}[]
    # With method = :points, the beliefs at which the value function is backed up (the
    # corners of the simplex first) and, for each number of periods left, an upper
    # bound on the value at each of them.
    points = Vector{T}[]
    upper_values = Vector{T}[]
    if method === :points
        local nstates = length(w0)
        append!(points, _simplex_grid(T, nstates, grid))
        local start = w0 ./ sum(w0)
        start in points || push!(points, start)
    end
    # An upper bound on the value of n periods from the weights w: the value function
    # is convex and scales with the weights, so it lies below the plane through the
    # corner values and below each "sawtooth" that hangs from one more point.
    function upper_at(n, w)
        n == 0 && return zero(T)
        local total = sum(w)
        iszero(total) && return zero(T)
        local b = w ./ total
        local values = upper_values[n]
        local nstates = length(b)
        local plane = sum(b[t] * values[t] for t in 1:nstates)
        local bound = plane
        for i in (nstates + 1):length(points)
            local q = points[i]
            local share = minimum(b[t] / q[t] for t in 1:nstates if !iszero(q[t]))
            local hanging = plane +
                            share * (values[i] - sum(q[t] * values[t] for t in 1:nstates))
            hanging < bound && (bound = hanging)
        end
        return total * bound
    end
    if by_vectors
        local nstates = length(w0)
        local same = Matrix{T}(I, nstates, nstates)
        local moved = [M === nothing ? same : Matrix{T}(M) for M in moves]
        local kernels = map(eachindex(actions)) do a
            local L = likelihoods[a]
            L === nothing && return [moved[a]]
            return [signal_of === :next ?
                    Diagonal(collect(T, view(L, z, :))) * moved[a] :
                    moved[a] * Diagonal(collect(T, view(L, z, :)))
                    for z in axes(L, 1)]
        end
        local G = zeros(T, 1, nstates)
        local H = zeros(T, 1, nstates)
        for n in 1:horizon
            if method === :points
                # A lower bound: at each point, the vector of the best action there
                # followed by the best of the vectors already held. Every vector is
                # the value of a plan that can be carried out.
                G, tags = _point_backup(G, Q, kernels, discount, points)
                push!(policy_vectors, (G, tags))
                # An upper bound at each point, from the bound for one period fewer.
                push!(upper_values,
                      [maximum(sum(view(Q, a, :) .* q) +
                               (n == 1 ? zero(T) :
                                discount * sum(upper_at(n - 1, next)
                                               for (_, next) in updates(q, a)))
                               for a in eachindex(actions)) for q in points])
                continue
            end
            G, tags = _backup_vectors(G, Q, kernels, discount)
            push!(policy_vectors, (G, tags))
            if moving
                H, plan_tags = _backup_vectors(H, Q, [[M] for M in moved], discount)
                push!(plan_vectors, (H, plan_tags))
            end
        end
    end
    # With method = :points the fixed plan is found exactly only where the state stays
    # put or moves in the same way whatever is done: then it is the best action for
    # the weights of each period in turn. Otherwise it is left out.
    shared_moves = moving && all(M -> M == first(moves), moves)
    plan_known = method !== :points || !moving || shared_moves
    vector_value(sets, n, w) = n == 0 ? zero(T) : maximum(first(sets[n]) * w)
    # The closures below recurse, so every variable they assign is declared local: a
    # name shared with the enclosing function would be overwritten by the inner calls.
    active = Dict{Tuple{Int,Vector{T}},Tuple{T,Int}}()
    function best(n, w)
        n == 0 && return (zero(T), 0)
        local key = (n, _weights_key(w))
        haskey(active, key) && return active[key]
        local scores = Q * w
        local value, choice = zero(T), 0
        for a in eachindex(actions)
            local v = scores[a]
            if n > 1
                for (_, next) in updates(w, a)
                    v += discount *
                         (by_vectors ? vector_value(policy_vectors, n - 1, next) :
                          first(best(n - 1, next)))
                end
            end
            (choice == 0 || v > value) && ((value, choice) = (v, a))
        end
        active[key] = (value, choice)
        return (value, choice)
    end
    # The best plan fixed in advance for n periods from the weights w, and its first
    # action. With a fixed state it is the best action for w, repeated.
    plans = Dict{Tuple{Int,Vector{T}},Tuple{T,Int}}()
    function plan(n, w)
        n == 0 && return (zero(T), 0)
        local scores = Q * w
        if !moving
            local first_best = argmax(scores)
            return (_horizon_weight(discount, n) * scores[first_best], first_best)
        end
        if method === :points
            local total, moved_on = maximum(scores), w
            for t in 1:(n - 1)
                moved_on = first(moves) * moved_on
                total += discount^t * maximum(Q * moved_on)
            end
            return (total, argmax(scores))
        end
        local key = (n, _weights_key(w))
        haskey(plans, key) && return plans[key]
        local value, choice = zero(T), 0
        for a in eachindex(actions)
            local v = scores[a]
            if n > 1
                local later = push_forward(w, a)
                v += discount *
                     (method === :vectors ? vector_value(plan_vectors, n - 1, later) :
                      first(plan(n - 1, later)))
            end
            (choice == 0 || v > value) && ((value, choice) = (v, a))
        end
        plans[key] = (value, choice)
        return (value, choice)
    end
    myopic = Dict{Tuple{Int,Vector{T}},T}()
    function replanned(n, w)
        n == 0 && return zero(T)
        local key = (n, _weights_key(w))
        haskey(myopic, key) && return myopic[key]
        local scores = Q * w
        local a = last(plan(n, w))
        local v = scores[a]
        if n > 1
            for (_, next) in updates(w, a)
                v += discount * replanned(n - 1, next)
            end
        end
        myopic[key] = v
        return v
    end
    function node(n, w, levels)
        local value, a = best(n, w)
        local total = sum(w)
        local branches = levels <= 1 || n <= 1 || likelihoods[a] === nothing ? [] :
                         [studies[a].signals[z] => node(n - 1, next, levels - 1)
                          for (z, next) in updates(w, a)]
        return (; period=horizon - n + 1, probability=total, belief=w ./ total,
                action=actions[a], value=value / total, branches)
    end

    no_learning, a0, fixed_plan = nothing, 0, nothing
    if plan_known
        no_learning, a0 = plan(horizon, w0)
        fixed_plan = Symbol[]
        let w = w0
            for n in horizon:-1:1
                a = last(plan(n, w))
                push!(fixed_plan, actions[a])
                w = push_forward(w, a)
            end
        end
    end
    # The state seen before every action: backward induction on the states.
    seen_value = zeros(T, length(w0))
    for _ in 1:horizon
        seen_value = [maximum(Q[a, t] +
                              discount * (moves[a] === nothing ? seen_value[t] :
                                          sum(moves[a][s, t] * seen_value[s]
                                              for s in eachindex(w0)))
                              for a in eachindex(actions)) for t in eachindex(w0)]
    end
    perfect = sum(w0 .* seen_value)
    active_value, first_choice_index = best(horizon, w0)
    passive && !plan_known &&
        throw(ArgumentError("with method = :points the passive manager is found only " *
                            "where the state stays put or moves in the same way " *
                            "whatever is done; use passive = false"))
    passive_value = passive ? replanned(horizon, w0) : nothing
    # With method = :points, `active` is a lower bound and this is the upper one.
    active_upper = active_value
    if method === :points
        active_upper = maximum(sum(view(Q, a, :) .* w0) +
                               (horizon == 1 ? zero(T) :
                                discount * sum(upper_at(horizon - 1, next)
                                               for (_, next) in updates(w0, a)))
                               for a in eachindex(actions))
    end
    base = problem.baseline === nothing ? nothing :
           findfirst(==(problem.baseline), actions)
    baseline_value = nothing
    if base !== nothing
        baseline_value = zero(T)
        let w = w0
            for t in 1:horizon
                baseline_value += discount^(t - 1) * sum(view(Q, base, :) .* w)
                w = push_forward(w, base)
            end
        end
    end
    return (; horizon, discount, no_learning, passive=passive_value, active=active_value,
            active_upper, perfect,
            value_of_learning=plan_known ? active_value - no_learning : nothing,
            value_of_probing=passive ? active_value - passive_value : nothing,
            evpi=plan_known ? perfect - no_learning : nothing,
            baseline=baseline_value,
            act_now_action=plan_known ? actions[a0] : nothing,
            first_action=actions[first_choice_index],
            plan=fixed_plan,
            policy=depth == 0 ? nothing : node(horizon, w0, depth),
            beliefs_evaluated=length(active),
            value_function=by_vectors ?
                           [(; actions=actions[tags], vectors=G)
                            for (G, tags) in policy_vectors] : nothing)
end
