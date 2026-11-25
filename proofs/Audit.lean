/-
Axiom audit: `lake env lean Audit.lean` (or `make audit`). Every theorem below must report at
most `propext`, `Classical.choice` and `Quot.sound`.
-/
import DecisionAnalysisProofs

#print axioms DecisionAnalysisProofs.smoke

-- Attained extrema and first-choice selection (`decide`, `first_choice`).
#print axioms DecisionAnalysisProofs.exists_eq_maxOver
#print axioms DecisionAnalysisProofs.exists_eq_minOver
#print axioms DecisionAnalysisProofs.exists_isBayes
#print axioms DecisionAnalysisProofs.mem_maximisers
#print axioms DecisionAnalysisProofs.firstArgmax_spec
#print axioms DecisionAnalysisProofs.lt_of_lt_firstArgmax
#print axioms DecisionAnalysisProofs.eq_firstArgmax
#print axioms DecisionAnalysisProofs.firstArgmax_congr_of_order
#print axioms DecisionAnalysisProofs.isBayes_bayesAction
#print axioms DecisionAnalysisProofs.isMinimaxRegret_minimaxRegretAction
#print axioms DecisionAnalysisProofs.isGammaMaximin_gammaMaximinAction
#print axioms DecisionAnalysisProofs.bayesAction_rescale

-- T2, T3: regret against expected utility.
#print axioms DecisionAnalysisProofs.expectedRegret_eq
#print axioms DecisionAnalysisProofs.minOver_expectedRegret_eq_evpi
#print axioms DecisionAnalysisProofs.IsBayes.expectedRegret_eq_evpi
#print axioms DecisionAnalysisProofs.isBayes_iff_expectedRegret_le
#print axioms DecisionAnalysisProofs.setOf_isBayes_eq

-- T7: positive affine rescaling.
#print axioms DecisionAnalysisProofs.isBayes_rescale
#print axioms DecisionAnalysisProofs.evpi_rescale
#print axioms DecisionAnalysisProofs.evsi_rescale
#print axioms DecisionAnalysisProofs.evppi_rescale
#print axioms DecisionAnalysisProofs.regret_rescale
#print axioms DecisionAnalysisProofs.isMinimaxRegret_rescale

-- T0, T6: garbling.
#print axioms DecisionAnalysisProofs.IsGarblingOf.refl
#print axioms DecisionAnalysisProofs.IsGarblingOf.trans
#print axioms DecisionAnalysisProofs.valueWithInformation_garble_le
#print axioms DecisionAnalysisProofs.IsGarblingOf.valueWithInformation_le
#print axioms DecisionAnalysisProofs.IsGarblingOf.evsi_le
#print axioms DecisionAnalysisProofs.valueWithInformation_comp_le
#print axioms DecisionAnalysisProofs.isGarblingOf_iff_exists_comp

-- T1, T5, T4, T14: value of information.
#print axioms DecisionAnalysisProofs.evpi_nonneg
#print axioms DecisionAnalysisProofs.evsi_nonneg
#print axioms DecisionAnalysisProofs.evsi_le_evpi
#print axioms DecisionAnalysisProofs.evsi_constant
#print axioms DecisionAnalysisProofs.evsi_identity
#print axioms DecisionAnalysisProofs.evppi_eq_evsi
#print axioms DecisionAnalysisProofs.evppi_nonneg
#print axioms DecisionAnalysisProofs.evppi_le_evpi
#print axioms DecisionAnalysisProofs.evppi_id
#print axioms DecisionAnalysisProofs.evppi_const
#print axioms DecisionAnalysisProofs.evppi_mono
#print axioms DecisionAnalysisProofs.evppi_mono_of_factorsThrough
#print axioms DecisionAnalysisProofs.evsi_eq_zero_iff
#print axioms DecisionAnalysisProofs.evsi_eq_zero_iff_of_pos

-- `bayes_invert` and the two forms of `value_with_information`.
#print axioms DecisionAnalysisProofs.posterior_normalised
#print axioms DecisionAnalysisProofs.posterior_nonneg
#print axioms DecisionAnalysisProofs.sum_signalProbability
#print axioms DecisionAnalysisProofs.sum_signalProbability_mul_posterior
#print axioms DecisionAnalysisProofs.signalValue_eq_posterior
#print axioms DecisionAnalysisProofs.signalValue_eq_zero
#print axioms DecisionAnalysisProofs.maxOver_signalValue_eq_zero
#print axioms DecisionAnalysisProofs.valueWithInformation_eq_posterior_form

-- T12: credal rules.
#print axioms DecisionAnalysisProofs.lowerExpectedUtility_le
#print axioms DecisionAnalysisProofs.IsGammaMaximin.gammaMaximinValue_le
#print axioms DecisionAnalysisProofs.gammaMaximinValue_le_actNowValue
#print axioms DecisionAnalysisProofs.IsBayes.isEAdmissible
#print axioms DecisionAnalysisProofs.IsEAdmissible.isMaximal
#print axioms DecisionAnalysisProofs.IsMaximal.isIntervalUndominated
#print axioms DecisionAnalysisProofs.IsGammaMaximin.isMaximal
#print axioms DecisionAnalysisProofs.IsMaximal.not_dominated_on_hull

-- T12b, easy direction: mixed dominance refutes E-admissibility.
#print axioms DecisionAnalysisProofs.IsMixedDominated.not_isEAdmissible
#print axioms DecisionAnalysisProofs.isMixedDominated_of_not_isMaximal

-- T10: total variation and the decision-loss bound (`decision_loss_bound`).
#print axioms DecisionAnalysisProofs.totalVariation_nonneg
#print axioms DecisionAnalysisProofs.totalVariation_eq_zero_iff
#print axioms DecisionAnalysisProofs.totalVariation_le_one
#print axioms DecisionAnalysisProofs.sum_sub_mul_le
#print axioms DecisionAnalysisProofs.abs_expectedUtility_sub_le
#print axioms DecisionAnalysisProofs.decision_loss_nonneg
#print axioms DecisionAnalysisProofs.decision_loss_le
#print axioms DecisionAnalysisProofs.IsBayes.decision_loss_le
#print axioms DecisionAnalysisProofs.decision_loss_bound_tight

-- T6a: deficiency bounds the value gap (`deficiency`).
#print axioms DecisionAnalysisProofs.IsDeficientWithin.mono
#print axioms DecisionAnalysisProofs.isDeficientWithin_zero_iff
#print axioms DecisionAnalysisProofs.IsGarblingOf.isDeficientWithin
#print axioms DecisionAnalysisProofs.valueWithInformation_le_add_of_totalVariation_le
#print axioms DecisionAnalysisProofs.IsDeficientWithin.valueWithInformation_le
#print axioms DecisionAnalysisProofs.IsDeficientWithin.evsi_le

-- T16: selection bias from maximising estimated payoffs.
#print axioms DecisionAnalysisProofs.actNowValue_meanTable_le
#print axioms DecisionAnalysisProofs.perfectValue_meanTable_le
#print axioms DecisionAnalysisProofs.valueWithInformation_meanTable_le
#print axioms DecisionAnalysisProofs.sum_mul_expectedUtility_meanTable_le
#print axioms DecisionAnalysisProofs.evpi_bias_sign_indefinite

-- T15: the EVPI envelope (`robust_evpi`, `evpi_envelope_model`).
#print axioms DecisionAnalysisProofs.perfectValue_mixture
#print axioms DecisionAnalysisProofs.actNowValue_mixture_le
#print axioms DecisionAnalysisProofs.sum_mul_evpi_le_evpi_mixture
#print axioms DecisionAnalysisProofs.minOver_evpi_le_evpi_mixture
#print axioms DecisionAnalysisProofs.exists_forall_evpi_le_evpi_mixture
#print axioms DecisionAnalysisProofs.EvpiEnvelopeFeasible.objective_le
#print axioms DecisionAnalysisProofs.evpiEnvelopeFeasible_actNowValue
#print axioms DecisionAnalysisProofs.evpiEnvelope_optimum
#print axioms DecisionAnalysisProofs.evpi_max_not_at_generating_point

-- Farkas and Ville: the duality behind the completeness of the "no" certificates.
#print axioms DecisionAnalysisProofs.farkas_cone
#print axioms DecisionAnalysisProofs.exists_mixed_pos_or_exists_mixed_nonpos

-- T12b and S1: completeness of the "no" certificates.
#print axioms DecisionAnalysisProofs.isEAdmissible_or_isMixedDominated
#print axioms DecisionAnalysisProofs.not_isEAdmissible_iff_isMixedDominated
#print axioms DecisionAnalysisProofs.isGarblingOf_of_forall_valueWithInformation_le
#print axioms DecisionAnalysisProofs.not_isGarblingOf_iff_exists_separating
#print axioms DecisionAnalysisProofs.isGarblingOf_iff_forall_valueWithInformation_le

-- T13: negative value of information under Gamma-maximin.
#print axioms DecisionAnalysisProofs.dilation_robust_evsi_negative

-- Survey first or act now (`survey_or_act`).
#print axioms DecisionAnalysisProofs.survey_first_iff
#print axioms DecisionAnalysisProofs.survey_first_tie
#print axioms DecisionAnalysisProofs.survey_same_action_lt
#print axioms DecisionAnalysisProofs.ratio_le_of_parametric

-- T8: minimax regret and independence of irrelevant alternatives.
#print axioms DecisionAnalysisProofs.isMinimaxRegret_iiaSmall_iff
#print axioms DecisionAnalysisProofs.isMinimaxRegret_iiaLarge_iff
#print axioms DecisionAnalysisProofs.minimaxRegret_violates_iia

-- T9: the worked examples.
#print axioms DecisionAnalysisProofs.Examples.floodPrior_nonneg
#print axioms DecisionAnalysisProofs.Examples.floodPrior_normalised
#print axioms DecisionAnalysisProofs.Examples.gauge_stochastic
#print axioms DecisionAnalysisProofs.Examples.flipKernel_stochastic
#print axioms DecisionAnalysisProofs.Examples.flood_expectedUtility
#print axioms DecisionAnalysisProofs.Examples.flood_actNowValue
#print axioms DecisionAnalysisProofs.Examples.flood_isBayes
#print axioms DecisionAnalysisProofs.Examples.flood_worstCase
#print axioms DecisionAnalysisProofs.Examples.flood_maxRegret
#print axioms DecisionAnalysisProofs.Examples.flood_isMinimaxRegret
#print axioms DecisionAnalysisProofs.Examples.flood_expectedRegret
#print axioms DecisionAnalysisProofs.Examples.flood_evpi
#print axioms DecisionAnalysisProofs.Examples.flood_evpi_eq_expectedRegret
#print axioms DecisionAnalysisProofs.Examples.flood_evsi_gauge
#print axioms DecisionAnalysisProofs.Examples.flood_evsi_garbled
#print axioms DecisionAnalysisProofs.Examples.flood_garbled_isGarblingOf
#print axioms DecisionAnalysisProofs.Examples.flood_ensemble_expectedUtility
#print axioms DecisionAnalysisProofs.Examples.flood_lowerExpectedUtility
#print axioms DecisionAnalysisProofs.Examples.flood_gammaMaximinValue
#print axioms DecisionAnalysisProofs.Examples.flood_isGammaMaximin
#print axioms DecisionAnalysisProofs.Examples.flood_isEAdmissible
#print axioms DecisionAnalysisProofs.Examples.epidemic_evpi
#print axioms DecisionAnalysisProofs.Examples.epidemic_evsi
#print axioms DecisionAnalysisProofs.Examples.epidemic_net
#print axioms DecisionAnalysisProofs.Examples.equipment_evpi
#print axioms DecisionAnalysisProofs.Examples.equipment_evsi
#print axioms DecisionAnalysisProofs.Examples.equipment_net
#print axioms DecisionAnalysisProofs.Examples.equipment_expectedRegret
#print axioms DecisionAnalysisProofs.Examples.equipment_maxRegret

-- Certificates and witnesses.
#print axioms DecisionAnalysisProofs.Certificate.mem_indicesWhere
#print axioms DecisionAnalysisProofs.Certificate.check_sound
#print axioms DecisionAnalysisProofs.Certificate.CoreMatches.bayesActions_iff
#print axioms DecisionAnalysisProofs.Certificate.CoreMatches.minimaxRegretActions_iff
#print axioms DecisionAnalysisProofs.Certificate.InfoMatches.evsi_bounds
#print axioms DecisionAnalysisProofs.Certificate.certificate_spec
#print axioms DecisionAnalysisProofs.Certificate.GarblingWitnessOk.isGarblingOf
#print axioms DecisionAnalysisProofs.Certificate.RawGarbling.accepts_sound
#print axioms DecisionAnalysisProofs.Certificate.EAdmissibleWitnessOk.isEAdmissible
#print axioms DecisionAnalysisProofs.Certificate.RawEAdmissible.accepts_sound
#print axioms DecisionAnalysisProofs.Certificate.MixedDominanceWitnessOk.not_isEAdmissible
#print axioms DecisionAnalysisProofs.Certificate.RawNotEAdmissible.accepts_sound
#print axioms DecisionAnalysisProofs.Certificate.RawNotGarbling.accepts_sound
