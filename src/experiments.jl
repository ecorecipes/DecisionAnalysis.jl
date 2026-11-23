# Experiments: information sources as likelihood matrices, Bayesian inversion, garbling,
# the garbling test and deficiency.

# `nothing` if every column of `L` is a probability vector within `atol`.
function _stochastic_defect(L::AbstractMatrix, atol::Real)
    for j in axes(L, 2)
        defect = _probability_defect(view(L, :, j), atol)
        defect === nothing || return "column $j $defect"
    end
    return nothing
end

"""
    Experiment(likelihood; signals, states=nothing, cost=0, atol)
    Experiment(k::FiniteKernel; cost=0, atol)

An information source over a finite state space. `likelihood[z, θ] = P(z | θ)`: rows
are signals and columns are states, so every **column** sums to one (exactly for
`Rational` entries, within `atol` otherwise); a matrix that is not column-stochastic
throws [`InvalidExperimentError`](@ref). `signals` labels the rows (default
`:z1, :z2, ...`). `states`, when given, labels the columns and is checked against the
payoff table of any problem the experiment is used with; without it only the number
of states is checked. `cost` is the price of running the experiment, in payoff units;
it is used only by [`enbs`](@ref).

The second form takes a kernel of FiniteKernels from states to signals; its
`kernel_matrix` is already laid out signals-by-states. Several-axis spaces are
labelled as in [`FiniteBelief`](@ref).

```jldoctest
julia> e = Experiment([4//5 1//5; 1//5 4//5]; signals=[:negative, :positive]);

julia> size(e.likelihood)
(2, 2)
```
"""
struct Experiment{T<:Real,C<:Real}
    signals::Vector{Symbol}
    states::Union{Nothing,Vector{Symbol}}
    likelihood::Matrix{T}
    cost::C
    function Experiment{T,C}(signals::Vector{Symbol},
                             states::Union{Nothing,Vector{Symbol}}, likelihood::Matrix{T},
                             cost::C; atol::Real=_default_atol(T)) where {T<:Real,C<:Real}
        isempty(likelihood) &&
            throw(InvalidExperimentError("an experiment needs at least one signal and " *
                                         "one state"))
        length(signals) == size(likelihood, 1) ||
            throw(InvalidExperimentError("$(length(signals)) signal labels for " *
                                         "$(size(likelihood, 1)) rows of the likelihood"))
        allunique(signals) ||
            throw(InvalidExperimentError("signal labels must be distinct, got $signals"))
        if states !== nothing
            length(states) == size(likelihood, 2) ||
                throw(InvalidExperimentError("$(length(states)) state labels for " *
                                             "$(size(likelihood, 2)) columns of the " *
                                             "likelihood"))
            allunique(states) ||
                throw(InvalidExperimentError("state labels must be distinct, got $states"))
        end
        defect = _stochastic_defect(likelihood, atol)
        defect === nothing ||
            throw(InvalidExperimentError("the likelihood must have rows = signals and " *
                                         "columns = states, each column a probability " *
                                         "vector, but $defect"))
        isfinite(cost) || throw(ArgumentError("the cost must be finite, got $cost"))
        return new{T,C}(signals, states, likelihood, cost)
    end
end

function Experiment(likelihood::AbstractMatrix{T};
                    signals=_default_labels("z", size(likelihood, 1)), states=nothing,
                    cost::Real=0, kw...) where {T<:Real}
    labels = states === nothing ? nothing : _labels(states)
    return Experiment{T,typeof(cost)}(_labels(signals), labels, Matrix{T}(likelihood),
                                      cost; kw...)
end

function _space_labels(X::FiniteSpace)
    return ndims(X) == 1 ? copy(only(factors(X)).labels) :
           [Symbol(join(s, "_")) for s in joint_states(X)]
end

function Experiment(k::FiniteKernel; cost::Real=0, kw...)
    (ndims(k.dom) >= 1 && ndims(k.codom) >= 1) ||
        throw(ArgumentError("an experiment needs a kernel from a state space to a " *
                            "signal space, both non-empty"))
    return Experiment(kernel_matrix(k); signals=_space_labels(k.codom),
                      states=_space_labels(k.dom), cost, kw...)
end

Base.eltype(::Type{<:Experiment{T}}) where {T} = T
Base.eltype(::Experiment{T}) where {T} = T

function Base.:(==)(a::Experiment, b::Experiment)
    return a.signals == b.signals && a.states == b.states &&
           a.likelihood == b.likelihood && a.cost == b.cost
end
function Base.hash(e::Experiment, h::UInt)
    return hash(e.cost, hash(e.likelihood, hash(e.states, hash(e.signals, hash(:E, h)))))
end

state_labels(e::Experiment) = e.states

"""
    signal_labels(e::Experiment) -> Vector{Symbol}

The signal labels of an experiment, in row order of its likelihood.
"""
signal_labels(e::Experiment) = e.signals

# The experiment's states must be the given ones (by label if it has labels, by count
# otherwise).
function _check_states(e::Experiment, states::Vector{Symbol}, what::AbstractString)
    if e.states === nothing
        size(e.likelihood, 2) == length(states) ||
            throw(InvalidExperimentError("the likelihood has $(size(e.likelihood, 2)) " *
                                         "columns for the $(length(states)) states of " *
                                         "the $what"))
    else
        e.states == states ||
            throw(LabelMismatchError("states of the experiment against the $what", states,
                                     e.states))
    end
    return nothing
end

"""
    random_experiment(rng, nz, n; T=Float64, signals, states=nothing, cost=0) -> Experiment

A random experiment with `nz` signals over `n` states, for tests and examples; each
column of the likelihood is drawn as in [`random_belief`](@ref).
"""
function random_experiment(rng::AbstractRNG, nz::Integer, n::Integer;
                           T::Type{<:Real}=Float64, signals=_default_labels("z", nz),
                           states=nothing, cost::Real=0)
    L = reduce(hcat, [_random_probabilities(rng, T, nz) for _ in 1:n])
    return Experiment(L; signals, states, cost)
end

# Bayesian inversion
# ------------------

"""
    BayesianInversion

What [`bayes_invert`](@ref) returns. Fields:

- `signals`, `states`: the labels.
- `joint`: the matrix `joint[z, θ] = p(θ) · L[z, θ]`.
- `marginal`: the signal marginal `P(z) = Σ_θ joint[z, θ]`.
- `posteriors`: one entry per signal, the [`FiniteBelief`](@ref) `P(θ | z)`, or
  `nothing` for a signal of probability zero, where the posterior is undefined.
"""
struct BayesianInversion{T<:Real}
    signals::Vector{Symbol}
    states::Vector{Symbol}
    joint::Matrix{T}
    marginal::Vector{T}
    posteriors::Vector{Union{Nothing,FiniteBelief{T}}}
end

"""
    bayes_invert(prior::FiniteBelief, experiment::Experiment) -> BayesianInversion
    bayes_invert(prior::AbstractVector, experiment::Experiment) -> BayesianInversion

Bayes' rule for a prior over states and an experiment: the signal marginal
`P(z) = Σ_θ p(θ) L[z, θ]` and, for every signal, the posterior
`P(θ | z) = p(θ) L[z, θ] / P(z)`.

A signal of probability zero has no posterior. Its entry in `posteriors` is
`nothing`; it is never `NaN` and never an arbitrary distribution. (The value of
information does not need the division at all; see [`value_with_information`](@ref).)

```jldoctest
julia> prior = FiniteBelief([:low, :high], [3 // 5, 2 // 5]);

julia> test = Experiment([4//5 1//5; 1//5 4//5]; signals=[:negative, :positive]);

julia> inv = bayes_invert(prior, test);

julia> inv.marginal
2-element Vector{Rational{Int64}}:
 14//25
 11//25

julia> probabilities(inv.posteriors[2])
2-element Vector{Rational{Int64}}:
 3//11
 8//11
```
"""
function bayes_invert(prior::FiniteBelief, experiment::Experiment)
    _check_states(experiment, prior.states, "prior")
    joint = experiment.likelihood .* permutedims(prior.probabilities)
    T = eltype(joint)
    marginal = vec(sum(joint; dims=2))
    posteriors = Union{Nothing,FiniteBelief{T}}[]
    for z in eachindex(marginal)
        if iszero(marginal[z])
            push!(posteriors, nothing)
        else
            # Normalise by the row's own sum so that rounding cannot fail the check.
            push!(posteriors, FiniteBelief(prior.states, joint[z, :] ./ marginal[z]))
        end
    end
    return BayesianInversion{T}(experiment.signals, prior.states, joint, marginal,
                                posteriors)
end

function bayes_invert(prior::AbstractVector{<:Real}, experiment::Experiment)
    states = experiment.states === nothing ? _default_labels("s", length(prior)) :
             experiment.states
    return bayes_invert(FiniteBelief(states, prior), experiment)
end

# Garbling
# --------

"""
    garble(experiment::Experiment, G; signals, cost=experiment.cost, atol) -> Experiment

The experiment obtained by passing the signal of `experiment` through the noisy
channel `G`, where `G[z′, z] = P(z′ | z)`: rows are the new signals, columns the old
ones, and every column sums to one. The likelihood of the result is `G * L`. By
Blackwell's theorem a garbled experiment is worth no more than the original to any
decision-maker with a single prior; see [`evsi`](@ref).

`G` that is not column-stochastic, or has the wrong number of columns, throws
[`InvalidExperimentError`](@ref).
"""
function garble(experiment::Experiment, G::AbstractMatrix{<:Real};
                signals=_default_labels("z", size(G, 1)), cost::Real=experiment.cost,
                atol::Real=_default_atol(promote_type(eltype(G), eltype(experiment))))
    size(G, 2) == length(experiment.signals) ||
        throw(InvalidExperimentError("the garbling matrix has $(size(G, 2)) columns for " *
                                     "the $(length(experiment.signals)) signals of the " *
                                     "experiment"))
    defect = isempty(G) ? "it is empty" : _stochastic_defect(G, atol)
    defect === nothing ||
        throw(InvalidExperimentError("the garbling matrix must have rows = new " *
                                     "signals and columns = old signals, each column a " *
                                     "probability vector, but $defect"))
    return Experiment(G * experiment.likelihood; signals, states=experiment.states, cost,
                      atol)
end

function _check_same_states(e::Experiment, f::Experiment)
    size(e.likelihood, 2) == size(f.likelihood, 2) ||
        throw(InvalidExperimentError("the experiments have $(size(e.likelihood, 2)) " *
                                     "and $(size(f.likelihood, 2)) states"))
    if e.states !== nothing && f.states !== nothing && e.states != f.states
        throw(LabelMismatchError("states of two experiments", e.states, f.states))
    end
    return nothing
end

# The deficiency programme: over column-stochastic G (rows = signals of `target`,
# columns = signals of `source`), minimise the largest ℓ¹ distance over states between
# column θ of G * L_source and column θ of L_target. Returns that minimum and G.
function _deficiency_lp(source::Experiment, target::Experiment)
    _check_same_states(source, target)
    L, Lt = source.likelihood, target.likelihood
    nz, S = size(L)
    nt = size(Lt, 1)
    # Variables: G[z′, z] (column-major), u[z′, θ] (column-major), d.
    g(zt, z) = (z - 1) * nt + zt
    u(zt, θ) = nt * nz + (θ - 1) * nt + zt
    nvar = nt * nz + nt * S + 1
    Aeq = zeros(_BigRat, nz, nvar)
    for z in 1:nz, zt in 1:nt
        Aeq[z, g(zt, z)] = 1
    end
    Aub = zeros(_BigRat, 2 * nt * S + S, nvar)
    bub = zeros(_BigRat, 2 * nt * S + S)
    row = 0
    for θ in 1:S, zt in 1:nt
        # (G L)[z′, θ] − u ≤ Lt[z′, θ]  and  −(G L)[z′, θ] − u ≤ −Lt[z′, θ]
        for sign in (1, -1)
            row += 1
            for z in 1:nz
                Aub[row, g(zt, z)] = sign * _BigRat(L[z, θ])
            end
            Aub[row, u(zt, θ)] = -1
            bub[row] = sign * _BigRat(Lt[zt, θ])
        end
    end
    for θ in 1:S
        row += 1
        for zt in 1:nt
            Aub[row, u(zt, θ)] = 1
        end
        Aub[row, nvar] = -1
    end
    c = zeros(_BigRat, nvar)
    c[nvar] = 1
    sol = _solve_lp(c, Aeq, ones(_BigRat, nz), Aub, bub)
    sol.status === :optimal || throw(LinearProgramError(:deficiency, sol.status))
    return sol.objective, reshape(sol.x[1:(nt * nz)], nt, nz)
end

"""
    deficiency(e::Experiment, e′::Experiment)

Le Cam's deficiency of `e` with respect to `e′`: how far `e` is from being able to
simulate `e′`,

    δ(e, e′) = min_G max_θ TV(G · L_e[·, θ], L_e′[·, θ]),

the minimum over column-stochastic matrices `G` (garblings of `e`) of the largest
total-variation distance, over states, between the garbled likelihood and the
likelihood of `e′`. Total variation is half the ℓ¹ distance, as in
[`total_variation`](@ref).

`δ(e, e′) = 0` exactly when `e′` is a garbling of `e`. It is not symmetric, and most
pairs of experiments have both `δ(e, e′) > 0` and `δ(e′, e) > 0`: they are
incomparable in Blackwell's order. The deficiency still bounds the value gap: for a
payoff table with range `B` (largest minus smallest payoff), the value of acting on
`e′` exceeds that of acting on `e` by at most `B δ(e, e′)`, for every prior (simulate
`e′` from `e` through the minimising `G` and act as if the result were `e′`).

Computed by an exact linear programme. The result is a `Rational{BigInt}` for
rational likelihoods and a float for floating-point ones.

```jldoctest
julia> e = Experiment([4//5 1//5; 1//5 4//5]);

julia> noisy = garble(e, [3//4 1//4; 1//4 3//4]);

julia> deficiency(e, noisy), deficiency(noisy, e)
(0//1, 3//20)
```
"""
function deficiency(e::Experiment, e′::Experiment)
    d, _ = _deficiency_lp(e, e′)
    return _from_big(promote_type(eltype(e), eltype(e′)), d / 2)
end

"""
    is_garbling_of(e′::Experiment, e::Experiment; atol) -> Union{Matrix,Nothing}

Whether `e′` is a garbling of `e`, that is, whether `e′` is less informative than `e`
in Blackwell's order. Returns a witness: a column-stochastic matrix `G` with
`G[z′, z] = P(z′ | z)` and `G * L_e = L_e′`, or `nothing` when there is none.

The test is the linear programme of [`deficiency`](@ref): a witness exists exactly
when `deficiency(e, e′)` is zero. `atol` is the deficiency below which a
floating-point pair counts as a garbling (default `1e-8`); for `Rational`
likelihoods the default is zero and the answer is exact, with a witness whose product
reproduces `L_e′` entry for entry.
"""
function is_garbling_of(e′::Experiment, e::Experiment;
                        atol::Real=_default_atol(promote_type(eltype(e), eltype(e′))))
    d, G = _deficiency_lp(e, e′)
    d / 2 <= atol || return nothing
    return _from_big.(promote_type(eltype(e), eltype(e′)), G)
end
