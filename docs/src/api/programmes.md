# Linear programmes, reports, certificates and exceptions

## Linear programmes

Every linear programme of the package is stated once, as a JuMP model, by an
exported builder. [`solve_program`](@ref) solves one by the built-in exact simplex
or by any JuMP optimiser; the functions above ([`decide`](@ref) with
[`EAdmissible`](@ref), [`is_garbling_of`](@ref), [`deficiency`](@ref),
[`robust_evpi`](@ref)) build, solve and verify.

```@docs
eadmissibility_model
garbling_model
deficiency_model
evpi_envelope_model
mixed_dominance_model
separating_payoff_model
DecisionProgram
solve_program
ProgramSolution
```

## Reports and certificates

```@docs
DecisionReport
export_decision_certificate
CERTIFICATE_SCHEMA
export_garbling_witness
GARBLING_WITNESS_SCHEMA
export_eadmissible_witness
NOT_EADMISSIBLE_WITNESS_SCHEMA
export_not_eadmissible_witness
NOT_GARBLING_WITNESS_SCHEMA
export_not_garbling_witness
EADMISSIBLE_WITNESS_SCHEMA
```

## Exceptions

```@docs
DecisionAnalysisError
InvalidBeliefError
InvalidPayoffError
InvalidExperimentError
LabelMismatchError
BeliefFormError
NoFeasibleActionError
InvalidBaselineError
MissingBaselineError
InexactCertificateError
WitnessError
LinearProgramError
ExactVerificationError
InvalidOutcomeError
MissingExtensionError
```
