# Tables.jl row interfaces for PayoffTable, RegretTable, OutcomeTable and VOIEstimate.
# Long-format tables are used for PayoffTable, RegretTable and OutcomeTable so that
# tables with thousands of draws (from a SampleBelief) stay type-stable and avoid wide
# NamedTuples with thousands of columns.

Tables.istable(::Type{<:PayoffTable}) = true
Tables.rowaccess(::Type{<:PayoffTable}) = true
function Tables.rows(t::PayoffTable)
    na, ns = size(t.Q)
    has_se = t.standard_errors !== nothing
    has_rep = t.replicates !== nothing
    if has_se && has_rep
        return [(; action=t.actions[a], state=t.states[s], payoff=t.Q[a, s],
                 standard_error=t.standard_errors[a, s], replicates=t.replicates[a, s])
                for a in 1:na for s in 1:ns]
    elseif has_se
        return [(; action=t.actions[a], state=t.states[s], payoff=t.Q[a, s],
                 standard_error=t.standard_errors[a, s])
                for a in 1:na for s in 1:ns]
    elseif has_rep
        return [(; action=t.actions[a], state=t.states[s], payoff=t.Q[a, s],
                 replicates=t.replicates[a, s])
                for a in 1:na for s in 1:ns]
    end
    return [(; action=t.actions[a], state=t.states[s], payoff=t.Q[a, s])
            for a in 1:na for s in 1:ns]
end

Tables.istable(::Type{<:RegretTable}) = true
Tables.rowaccess(::Type{<:RegretTable}) = true
function Tables.rows(t::RegretTable)
    na, ns = size(t.R)
    return [(; action=t.actions[a], state=t.states[s], regret=t.R[a, s])
            for a in 1:na for s in 1:ns]
end

Tables.istable(::Type{<:OutcomeTable}) = true
Tables.rowaccess(::Type{<:OutcomeTable}) = true
function Tables.rows(t::OutcomeTable)
    na, nd, nr, no = size(t.values)
    col_names = (:draw, :action, :replicate, :status, t.outcomes...)
    out = NamedTuple{col_names}[]
    for d in 1:nd, a in 1:na, r in 1:nr
        st = t.status[a, d, r]
        st == _ABSENT && continue
        vals = (t.draws[d], t.actions[a], t.replicates[r], st,
                (t.values[a, d, r, k] for k in 1:no)...)
        push!(out, NamedTuple{col_names}(vals))
    end
    return out
end

Tables.istable(::Type{<:VOIEstimate}) = true
Tables.rowaccess(::Type{<:VOIEstimate}) = true
function Tables.rows(e::VOIEstimate)
    return [(; quantity=e.quantity, method=e.method, value=e.value,
             standard_error=e.standard_error, bias_act_now=e.bias.act_now,
             bias_cells=e.bias.cells)]
end
