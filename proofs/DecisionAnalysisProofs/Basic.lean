import FiniteKernelsProofs.Finite.Kernel

/-!
# DecisionAnalysisProofs

Lean 4 / Mathlib formalisation accompanying `DecisionAnalysis.jl` (toolchain
`leanprover/lean4:v4.30.0`, Mathlib tag `v4.30.0`).

**Purpose.** These proofs back the Julia code and its certificates. They give each exported
quantity a definition that Julia's exact (`Rational`) results can be compared with, prove the
properties the Julia test suite asserts on random problems, and prove sound the checkers that
accept or reject what Julia exports (decision certificates, and the witnesses returned by its
linear programmes). They are not a development of decision theory: a result is here because a
Julia function, a property test, a certificate or an API decision rests on it.

Every library module is built by `lake build --wfail`; `Audit.lean` prints the axioms of the
headline results (only `propext`, `Classical.choice`, `Quot.sound`). `Roadmap.lean` lists
what remains and has no unproved declarations.

The project depends by path on `FiniteKernels.jl/proofs` (Lake package
`finite_kernels_proofs`); at `K = ℝ` an experiment is a `FiniteKernelsProofs.Finite.Kernel`
and garbling is `Kernel.comp` (`Theory/FiniteKernels.lean`).

## The model

Finite types of actions, states and signals; a payoff table `u : A → Θ → K`; a prior
`p : Θ → K`; an experiment `e : Θ → Z → K` with `e θ z = P(z | θ)` (the Julia likelihood
matrix is `likelihood[z, θ]`). `K` is any linearly ordered field, so the same definitions
serve the theorems (any `K`, in particular `ℝ`) and the computations (`K = ℚ`: worked
examples and the certificate checker, which runs these definitions and nothing else).
Nonnegativity and normalisation are separate predicates, assumed only where needed.

## By Julia function

| Julia (`DecisionAnalysis.jl`) | Lean definition | Lean results |
|:--|:--|:--|
| `act_now_value`, `decide(p, MaxExpectedUtility())` (`src/voi.jl`, `src/rules.jl`) | `expectedUtility`, `actNowValue`, `IsBayes`, `maximisers` | `exists_isBayes`, `isBayes_iff`, `mem_maximisers` |
| `first_choice` (`src/rules.jl`) | `firstArgmax`, `firstArgmin`, `bayesAction`, `minimaxRegretAction`, `gammaMaximinAction` | `firstArgmax_spec`, `lt_of_lt_firstArgmax`, `eq_firstArgmax`, `firstArgmax_congr_of_order`, `bayesAction_rescale` |
| `value_with_information`, `evsi`, `information_can_change_decision` (`src/voi.jl`) | `signalValue`, `valueWithInformation`, `evsi` | `evsi_nonneg`, `evsi_le_evpi`, `evsi_constant`, `evsi_identity`, `evsi_eq_zero_iff`, `evsi_eq_zero_iff_of_pos`, `valueWithInformation_eq_posterior_form` |
| `evpi` (`src/voi.jl`) | `perfectValue`, `evpi` | `evpi_nonneg`, `minOver_expectedRegret_eq_evpi` |
| `evppi(problem, partition)` (`src/voi.jl`) | `partialValue`, `evppi`, `deterministic` | `evppi_eq_evsi`, `evppi_nonneg`, `evppi_le_evpi`, `evppi_mono`, `evppi_mono_of_factorsThrough`, `evppi_id`, `evppi_const` |
| `regret_table`, `expected_regret`, `max_regret`, `decide(p, MinimaxRegret())` (`src/regret.jl`, `src/rules.jl`) | `regret`, `expectedRegret`, `maxRegret`, `IsMinimaxRegret` | `expectedRegret_eq`, `isBayes_iff_expectedRegret_le`, `regret_rescale`, `isMinimaxRegret_rescale`, `minimaxRegret_violates_iia` |
| `bayes_invert` (`src/experiments.jl`) | `signalProbability`, `posterior` | `posterior_normalised`, `posterior_nonneg`, `sum_signalProbability`, `sum_signalProbability_mul_posterior`, `signalValue_eq_zero` |
| `garble`, `is_garbling_of`, `garbling_result` (`src/experiments.jl`; programme `garbling_model` in `src/programs.jl`) | `garble`, `IsGarblingOf` | `IsGarblingOf.refl`, `IsGarblingOf.trans`, `valueWithInformation_garble_le`, `IsGarblingOf.evsi_le`, `Certificate.RawGarbling.accepts_sound` |
| `decide(p, GammaMaximin())`, `EAdmissible()`, `Maximal()`, `IntervalDominance()` (`src/rules.jl`; programme `eadmissibility_model` in `src/programs.jl`) | `lowerExpectedUtility`, `gammaMaximinValue`, `IsGammaMaximin`, `IsEAdmissible`, `IsMaximal`, `IsIntervalUndominated` | `IsGammaMaximin.gammaMaximinValue_le`, `IsBayes.isEAdmissible`, `IsEAdmissible.isMaximal`, `IsMaximal.isIntervalUndominated`, `IsMaximal.not_dominated_on_hull`, `IsGammaMaximin.isMaximal`, `Certificate.RawEAdmissible.accepts_sound` |
| payoff rescaling (`PayoffTable(c .* Q .+ d)`, `c * table`) | `rescale` | `isBayes_rescale`, `bayesAction_rescale`, `evpi_rescale`, `evsi_rescale`, `evppi_rescale` |
| `total_variation`, `decision_loss_bound` (`src/fidelity.jl`) | `totalVariation` | `abs_expectedUtility_sub_le`, `IsBayes.decision_loss_le`, `decision_loss_le`, `decision_loss_bound_tight` |
| `deficiency` (`src/experiments.jl`; programme `deficiency_model`) | `IsDeficientWithin` | `IsDeficientWithin.valueWithInformation_le`, `isDeficientWithin_zero_iff` |
| `robust_evpi` (`src/voi.jl`; programme `evpi_envelope_model`) | `mixture`, `EvpiEnvelopeFeasible`, `evpiEnvelopeObjective` | `minOver_evpi_le_evpi_mixture`, `evpiEnvelope_optimum`, `evpi_max_not_at_generating_point` |
| `robust_evsi` (`src/voi.jl`), Γ-maximin | `updatedLowerUtility`, `strategyPayoff`, `strategyLowerValue` | `dilation_robust_evsi_negative` |
| a payoff table estimated from replicates (docstring of `evpi`) | `meanTable` | `actNowValue_meanTable_le`, `sum_mul_expectedUtility_meanTable_le`, `evpi_bias_sign_indefinite` |
| negative answers of `decide(p, EAdmissible())` and `garbling_result` | `IsMixedDominated`, `Certificate.RawNotEAdmissible`, `Certificate.RawNotGarbling` | `Certificate.RawNotEAdmissible.accepts_sound`, `Certificate.RawNotGarbling.accepts_sound`, `not_isEAdmissible_iff_isMixedDominated`, `not_isGarblingOf_iff_exists_separating` |
| `export_decision_certificate` (`src/certificates.jl`) | `Certificate.Raw`, `Certificate.check` | `Certificate.check_sound`, `Certificate.certificate_spec` |
| `export_garbling_witness`, `export_eadmissible_witness` (`src/certificates.jl`; any `optimizer`, verified exactly before writing) | `Certificate.RawGarbling`, `Certificate.RawEAdmissible` | `Certificate.RawGarbling.accepts_sound`, `Certificate.RawEAdmissible.accepts_sound` |
| `fsd_dominates`, `ssd_dominates`, `StochasticDominance` (`src/deep_uncertainty.jl`) | `FSD`, `StrictFSD`, `SSD`, `StrictSSD`, `IsFSDUndominated`, `IsSSDUndominated` | `expected_le_of_fsd`, `expected_le_of_ssd`, `cvarObj_le_of_ssd`, `ssd_of_fsd`, `IsSSDUndominated.isFSDUndominated`, `exists_isFSDUndominated`, `exists_isSSDUndominated` |
| `info_gap_robustness`, `info_gap_opportuneness` (`src/deep_uncertainty.jl`) | `InTVBall`, `GuaranteedAt`, `AchievableAt`, `ShiftFeasible` | `guaranteedAt_antitone`, `achievableAt_monotone`, `guaranteedAt_zero_iff`, `guaranteedAt_one_iff`, `tvBall_drop_le`, `exists_tvBall_attaining_shift`, `greedy_drop_shift_optimal`, `greedy_gain_shift_optimal` |
| `ce_evpi`, `ce_evppi`, `ce_evsi` (`src/ce_voi.jl`), `SmoothAmbiguity` (`src/aggregators.jl`) | `caraCE`, `caraInformedVal`, `smoothAmbiguity`, `expSmoothAmbiguity` | `ceEvsi_linear_eq_evsi`, `caraCE_logSumExp_eq`, `caraCE_le_mean`, `caraCE_concave`, `cara_buying_price_eq_ce_gain`, `lowerExpectedUtility_le_expSmoothAmbiguity`, `expSmoothAmbiguity_le_mixture`, `expSmoothAmbiguity_concave`, `smoothAmbiguity_le_mixture_of_concave` |
| `evpim`, `evsim`, `evmi` (`src/voi.jl`), `prune_generators`, `constrained_policy_model` (`src/programs.jl`) | `evpim`, `evsim`, `evmiEnsemble`, `evmiPair`, `IsRedundantGenerator`, `ConstrainedPolicyFeasible` | `evpim_nonneg`, `evsim_le_evpim`, `evim_with_experiment_eq_add`, `evmiEnsemble_nonneg`, `evmiEnsemble_eq_zero_iff`, `prune_redundant_isEAdmissible`, `constrained_policy_le_dual` |
| `adaptive_management` with time-varying `payoffs`, `experiments`, `transitions` and belief normalisation (`src/adaptive.jl`) | `tvDynamicValue`, `tvDynamicPolicyValue`, `tvOpenLoopValue`, `tvObservedValue` | `tvOpenLoopValue_le_tvDynamicPolicyValue`, `tvDynamicPolicyValue_le_tvDynamicValue`, `tvDynamicValue_le_sum_tvObservedValue`, `tvDynamicValue_const_eq_dynamicValue`, `tvDynamicValue_smul`, `tvDynamicValue_eq_total_mul_normalised` |
| `test/fixtures/worked_examples.json` | `Examples.floodU`, `Examples.epidemicU`, `Examples.equipmentU`, ... | `Examples.flood_evpi`, `Examples.flood_evsi_gauge`, `Examples.flood_evsi_garbled`, ... |

## Scope

Exact finite-model results, not a verification of Julia's floating-point arrays. The Lean and
Julia definitions are separate hand-written texts, checked to agree by the certificate
checker on exact problems and by the shared fixtures; no theorem says that the Julia code
executes these definitions.
-/

namespace DecisionAnalysisProofs

/-- Smoke theorem so that the axiom audit always has a first line. -/
theorem smoke : (1 : Nat) + 1 = 2 := rfl

end DecisionAnalysisProofs
