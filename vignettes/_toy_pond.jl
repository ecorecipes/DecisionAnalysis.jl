# A toy stochastic model of a fungal disease in a pond of newts, used by the
# sample-based vignettes. All numbers are illustrative. The model is reached by
# DecisionAnalysis through four adapter verbs only.

using DecisionAnalysis
using Random: AbstractRNG

"""
A pond: `newts` animals, one of them infected at the start; `contact` is the weekly
transmission rate, `open` says whether the pond is unshaded (warm water clears
infections faster), and `action` is the management applied in week 3.
"""
Base.@kwdef struct ToyPond
    newts::Int = 60
    contact::Float64 = 0.5
    open::Bool = false
    action::Symbol = :none
    weeks::Int = 12
end

# Uncertain parameters come from one row of the draws.
function DecisionAnalysis.with_parameters(pond::ToyPond, θ)
    return ToyPond(; newts=θ.newts, contact=θ.contact, open=θ.habitat == "open",
                   action=pond.action, weeks=pond.weeks)
end

# An action is an intervention in week 3:
#   :none    no intervention (the baseline);
#   :treat   catch 70% of the newts and bathe them in antifungal: an infected newt is
#            cured, and handling kills 8% of the newts caught;
#   :remove  catch 70% of the newts and remove those that test positive: every
#            infected newt caught, and 12% of the uninfected ones (false positives).
function DecisionAnalysis.apply_action(pond::ToyPond, action::Symbol)
    action in (:none, :treat, :remove) || throw(ArgumentError("unknown action $action"))
    return ToyPond(; newts=pond.newts, contact=pond.contact, open=pond.open, action,
                   weeks=pond.weeks)
end

binomial(rng::AbstractRNG, n::Integer, p::Real) = count(_ -> rand(rng) < p, 1:n)

# One run: a chain-binomial epidemic in weekly steps. Every random number comes from
# `rng`, so two actions run with the same generator see the same epidemic until week 3.
function DecisionAnalysis.simulate(pond::ToyPond; rng::AbstractRNG)
    S, I = pond.newts - 1, 1
    handled = 0
    for week in 1:(pond.weeks)
        if week == 3 && pond.action !== :none
            caught_S, caught_I = binomial(rng, S, 0.7), binomial(rng, I, 0.7)
            handled = caught_S + caught_I
            if pond.action === :treat
                dead_S, dead_I = binomial(rng, caught_S, 0.08),
                                 binomial(rng, caught_I, 0.08)
                S += (caught_I - dead_I) - dead_S
                I -= caught_I
            else
                S -= binomial(rng, caught_S, 0.12)
                I -= caught_I
            end
        end
        N = S + I
        N == 0 && break
        infections = binomial(rng, S, 1 - exp(-pond.contact * I / N))
        deaths = binomial(rng, I, 0.25)
        recoveries = binomial(rng, I - deaths, pond.open ? 0.35 : 0.05)
        S += recoveries - infections
        I += infections - deaths - recoveries
    end
    return (; uninfected=S, infected=I, handled)
end

# The named outcomes of a run: counts of animals, not utilities.
function DecisionAnalysis.outcome(run::NamedTuple)
    return (; uninfected=run.uninfected,
            infected=run.infected,
            alive=run.uninfected + run.infected,
            handled=run.handled)
end

"Draws of the uncertain parameters: an illustrative prior, not an estimate."
function toy_draws(rng::AbstractRNG, n::Integer)
    return (; contact=exp.(log(0.45) .+ 0.7 .* randn(rng, n)), newts=rand(rng, 30:120, n),
            habitat=[rand(rng) < 0.35 ? "open" : "shaded" for _ in 1:n])
end

const TOY_ACTIONS = [:none, :treat, :remove]
