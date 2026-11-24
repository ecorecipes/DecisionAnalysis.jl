# The verbs a model package implements so that its simulator can be driven by
# `outcomes`. They are declared here with no methods for foreign types: a model package
# never depends on DecisionAnalysis, so the methods live in a weak-dependency extension
# on this side (or in the user's script).

"""
    with_parameters(prob, θ)

Adapter verb, **to be given methods for a model's problem type**: the problem `prob`
with its uncertain parameters set to the draw `θ`, a named tuple holding one row of
the parameter draws. It must not mutate `prob`. The package defines no method.

The four verbs are used by [`outcomes`](@ref) in this order: with
`p = apply_action(with_parameters(prob, θ), a)`, a run is `outcome(p, simulate(p; rng))`,
and `outcome(p, solution)` is `outcome(solution)` unless a method says otherwise.
"""
function with_parameters end

"""
    apply_action(prob, action)

Adapter verb, to be given methods for a model's problem type: the problem `prob`
under the intervention `action`. An action is an intervention on the model (a changed
rate, a pulse on a given day, a control switched on), and the method documents how
each action changes it; no intervention must be an action too, so that every analysis
has its baseline. It must not mutate `prob`. The package defines no method.
"""
function apply_action end

"""
    simulate(prob; rng)

Adapter verb, to be given methods for a model's problem type: one run of the model,
drawing every random number from `rng`. [`outcomes`](@ref) passes a
`Random.Xoshiro` seeded identically for every action at a given draw and replicate
(common random numbers), so a deterministic model can ignore `rng` and a stochastic
one must use nothing else. An individual-based model keeps runs comparable for
longer after an intervention if it derives one stream per individual from `rng`
instead of drawing from a single stream in event order. The package defines no
method.
"""
function simulate end

"""
    outcome(solution) -> NamedTuple
    outcome(prob, solution) -> NamedTuple

Adapter verb, to be given methods for a model's solution type: the named outcomes of
one run, a named tuple of finite real numbers (for example
`(; uninfected=41, infected=3)`). Outcomes are counts or other quantities of the
model, not utilities: the utility is applied afterwards by [`payoff_table`](@ref).
The package defines no method of the first form.

[`outcomes`](@ref) calls the second form, with the problem that was simulated (after
[`with_parameters`](@ref) and [`apply_action`](@ref)); its fallback is
`outcome(solution)`. Give the second form a method **for the problem type** when the
outcomes to read off depend on the problem (a focal species, a horizon, a threshold),
or when the solution is of a type the model package does not own (a named tuple, a
trajectory or solution type of another package): dispatch is then on the problem, and
no method is added for a foreign type.
"""
function outcome end

outcome(prob, solution) = outcome(solution)

"""
    parameter_draws(source)

The parameter draws of `source` as a Tables.jl column table (a named tuple of
vectors, one per named parameter, one entry per draw). Defined for a
[`SampleBelief`](@ref) and an [`OutcomeTable`](@ref); an adapter verb for anything
else that holds draws (a posterior sample, a design), to be given methods for it.
"""
function parameter_draws end

"""
    draw_weights(source)

The weights of the draws of `source`, normalised to sum to one. Defined for a
[`SampleBelief`](@ref) and an [`OutcomeTable`](@ref); an adapter verb for anything
else that holds draws. Equal weights are the default for a posterior sample; a
design that is not a posterior needs weights that say what it stands for.
"""
function draw_weights end

parameter_draws(b::SampleBelief) = b.draws
draw_weights(b::SampleBelief) = b.weights
