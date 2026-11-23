import DecisionAnalysisProofs.Finite.Garbling

/-!
# DecisionAnalysisProofs.Finite.ValueOfInformation

Bounds on the value of information, as corollaries of **T6** (`src/voi.jl`: `evpi`, `evppi`,
`evsi`). The two extreme experiments are `constantExperiment` (signal space `Unit`, tells
nothing; its value is the act-now value) and `deterministic id` (reveals the state; for a
nonnegative prior its value is the perfect-information value).

* **T1**, **`evpi_nonneg`**: `0 ≤ EVPI` for a nonnegative prior.
* **T5**, **`evsi_nonneg`** (normalised experiment) and **`evsi_le_evpi`** (nonnegative prior,
  stochastic experiment on the same state): `0 ≤ EVSI ≤ EVPI`. Also `evsi_constant`
  (an uninformative experiment is worth nothing) and `evsi_identity` (the identity experiment
  is worth the EVPI).
* **T4**, **`evppi_nonneg`**, **`evppi_le_evpi`**, **`evppi_mono`** and
  **`evppi_mono_of_factorsThrough`**: `0 ≤ EVPPI ≤ EVPI`, and if what is learned through `f'`
  is a function of what is learned through `f` (`f' = h ∘ f`, or `Function.FactorsThrough`)
  then `EVPPI f' ≤ EVPPI f`. Also `evppi_id` and `evppi_const`.
* **T14**, **`evsi_eq_zero_iff`**: for a normalised experiment, `EVSI = 0` iff one action
  maximises the joint-form objective `signalValue` after every signal.
  **`evsi_eq_zero_iff_of_pos`**: for a nonnegative prior and a stochastic experiment, iff one
  action is optimal after every signal of positive probability. (After such a signal,
  maximising the joint-form objective is maximising posterior expected utility, the two
  differing by the positive factor `signalProbability p e z`; that rescaling is not restated
  here because posteriors are not defined in this library.)

Not claimed: any of these for sampled or regression estimators of EVPPI or EVSI, and nothing
for imprecise beliefs, where free information can have negative value (T13,
`Finite/Dilation.lean`).
-/

set_option autoImplicit false
-- One ordered-field context is declared per file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ Z C C' : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z] [Fintype C]
  [Fintype C'] {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-- The experiment that tells nothing: one signal, whatever the state. -/
def constantExperiment : Θ → Unit → K := fun _ _ => 1

theorem stochasticKernel_constantExperiment :
    StochasticKernel (constantExperiment : Θ → Unit → K) :=
  ⟨fun _ _ => zero_le_one, fun _ => by simp [constantExperiment]⟩

/-- Deciding after the uninformative experiment is deciding now. -/
theorem valueWithInformation_constant (u : A → Θ → K) (p : Θ → K) :
    valueWithInformation u p constantExperiment = actNowValue u p := by
  simp only [valueWithInformation, actNowValue, Finset.univ_unique, Finset.sum_singleton]
  refine congrArg maxOver (funext fun a => ?_)
  simp [signalValue, constantExperiment, expectedUtility]

/-- Deciding after the identity experiment is deciding with the state known. -/
theorem valueWithInformation_identity [DecidableEq Θ] (u : A → Θ → K) {p : Θ → K}
    (hp : NonnegPrior p) :
    valueWithInformation u p (deterministic id) = perfectValue u p := by
  unfold valueWithInformation perfectValue
  refine Finset.sum_congr rfl fun z _ => ?_
  rw [← maxOver_mul_left _ (hp z)]
  refine congrArg maxOver (funext fun a => ?_)
  simp [signalValue, deterministic]

/-- Every normalised experiment garbles to the uninformative one. -/
theorem garble_constant {e : Θ → Z → K} (he : NormalisedKernel e) :
    garble e constantExperiment = constantExperiment := by
  funext θ x
  have h := he θ
  simpa [garble, constantExperiment] using h

/-- **T5.** EVSI is nonnegative for a normalised experiment. -/
theorem evsi_nonneg (u : A → Θ → K) (p : Θ → K) {e : Θ → Z → K} (he : NormalisedKernel e) :
    0 ≤ evsi u p e := by
  have h := valueWithInformation_garble_le u p e
    (stochasticKernel_constantExperiment (Θ := Z) (K := K))
  rw [garble_constant he, valueWithInformation_constant] at h
  unfold evsi
  linarith

/-- **T5.** EVSI is at most EVPI, for a nonnegative prior and a stochastic experiment on the
same state. -/
theorem evsi_le_evpi [DecidableEq Θ] (u : A → Θ → K) {p : Θ → K} (hp : NonnegPrior p)
    {e : Θ → Z → K} (he : StochasticKernel e) : evsi u p e ≤ evpi u p := by
  have h := valueWithInformation_garble_le u p (deterministic id) he
  rw [identity_garble, valueWithInformation_identity u hp] at h
  unfold evsi evpi
  linarith

/-- An uninformative experiment has EVSI zero. -/
theorem evsi_constant (u : A → Θ → K) (p : Θ → K) : evsi u p constantExperiment = 0 := by
  unfold evsi
  rw [valueWithInformation_constant, sub_self]

/-- The identity experiment has EVSI equal to EVPI. -/
theorem evsi_identity [DecidableEq Θ] (u : A → Θ → K) {p : Θ → K} (hp : NonnegPrior p) :
    evsi u p (deterministic id) = evpi u p := by
  unfold evsi evpi
  rw [valueWithInformation_identity u hp]

/-- **T1.** EVPI is nonnegative for a nonnegative prior. -/
theorem evpi_nonneg (u : A → Θ → K) {p : Θ → K} (hp : NonnegPrior p) : 0 ≤ evpi u p := by
  classical
  rw [← evsi_identity u hp]
  exact evsi_nonneg u p (normalisedKernel_deterministic id)

/-- **T4.** EVPPI is nonnegative. -/
theorem evppi_nonneg [DecidableEq C] (u : A → Θ → K) (p : Θ → K) (f : Θ → C) :
    0 ≤ evppi u p f := by
  rw [evppi_eq_evsi]
  exact evsi_nonneg u p (normalisedKernel_deterministic f)

/-- **T4.** EVPPI is at most EVPI for a nonnegative prior. -/
theorem evppi_le_evpi [DecidableEq C] (u : A → Θ → K) {p : Θ → K} (hp : NonnegPrior p)
    (f : Θ → C) : evppi u p f ≤ evpi u p := by
  classical
  rw [evppi_eq_evsi]
  exact evsi_le_evpi u hp (stochasticKernel_deterministic f)

/-- Learning the whole state is perfect information. -/
theorem evppi_id [DecidableEq Θ] (u : A → Θ → K) {p : Θ → K} (hp : NonnegPrior p) :
    evppi u p id = evpi u p := by
  rw [evppi_eq_evsi, evsi_identity u hp]

/-- Learning nothing (the one-cell partition) is worth nothing. -/
theorem evppi_const (u : A → Θ → K) (p : Θ → K) : evppi u p (fun _ => ()) = 0 := by
  have h : (deterministic (fun _ : Θ => ()) : Θ → Unit → K) = constantExperiment := by
    funext θ x
    simp [deterministic, constantExperiment]
  rw [evppi_eq_evsi, h, evsi_constant]

/-- A coarsening of a deterministic experiment is a garbling of it. -/
theorem deterministic_comp [DecidableEq C] [DecidableEq C'] (f : Θ → C) (h : C → C') :
    (deterministic (h ∘ f) : Θ → C' → K) = garble (deterministic f) (deterministic h) := by
  funext θ c'
  simp only [garble, deterministic]
  rw [Finset.sum_eq_single (f θ)]
  · simp
  · intro c _ hc
    simp [hc]
  · simp

/-- **T4.** EVPPI is monotone under refinement: learning `h (f θ)` is worth no more than
learning `f θ`. -/
theorem evppi_mono [DecidableEq C] [DecidableEq C'] (u : A → Θ → K) (p : Θ → K) (f : Θ → C)
    (h : C → C') : evppi u p (h ∘ f) ≤ evppi u p f := by
  rw [evppi_eq_evsi, evppi_eq_evsi, deterministic_comp]
  exact IsGarblingOf.evsi_le ⟨deterministic h, stochasticKernel_deterministic h, rfl⟩ u p

/-- **T4**, with refinement expressed as `Function.FactorsThrough`: if `f'` is constant on the
fibres of `f`, then `EVPPI f' ≤ EVPPI f`. -/
theorem evppi_mono_of_factorsThrough [DecidableEq C] [DecidableEq C'] [Nonempty C']
    (u : A → Θ → K) (p : Θ → K) {f : Θ → C} {f' : Θ → C'} (h : Function.FactorsThrough f' f) :
    evppi u p f' ≤ evppi u p f := by
  obtain ⟨k, rfl⟩ := (Function.factorsThrough_iff f').mp h
  exact evppi_mono u p f k

/-- **T14.** For a normalised experiment, EVSI vanishes iff a single action maximises the
joint-form objective after every signal. -/
theorem evsi_eq_zero_iff (u : A → Θ → K) (p : Θ → K) {e : Θ → Z → K}
    (he : NormalisedKernel e) :
    evsi u p e = 0 ↔ ∃ a, ∀ z a', signalValue u p e z a' ≤ signalValue u p e z a := by
  constructor
  · intro h
    obtain ⟨a, ha⟩ := exists_eq_maxOver (expectedUtility u p)
    refine ⟨a, ?_⟩
    have hsum : ∑ z, (maxOver (signalValue u p e z) - signalValue u p e z a) = 0 := by
      rw [Finset.sum_sub_distrib, sum_signalValue u p he]
      unfold evsi valueWithInformation actNowValue at h
      linarith
    have hz := (Finset.sum_eq_zero_iff_of_nonneg
      (fun z _ => sub_nonneg.mpr (le_maxOver (signalValue u p e z) a))).mp hsum
    intro z a'
    have h1 := hz z (Finset.mem_univ z)
    have h2 := le_maxOver (signalValue u p e z) a'
    linarith
  · rintro ⟨a, ha⟩
    have h1 : valueWithInformation u p e = expectedUtility u p a := by
      rw [← sum_signalValue u p he]
      exact Finset.sum_congr rfl fun z _ => maxOver_eq_of a rfl (ha z)
    have h2 := expectedUtility_le_actNowValue u p a
    have h3 := evsi_nonneg u p he
    unfold evsi at h3 ⊢
    linarith

/-- **T14**, in the usual wording: for a nonnegative prior and a stochastic experiment, EVSI
vanishes iff a single action is optimal after every signal of positive probability. -/
theorem evsi_eq_zero_iff_of_pos (u : A → Θ → K) {p : Θ → K} (hp : NonnegPrior p)
    {e : Θ → Z → K} (he : StochasticKernel e) :
    evsi u p e = 0 ↔ ∃ a, ∀ z, 0 < signalProbability p e z →
      ∀ a', signalValue u p e z a' ≤ signalValue u p e z a := by
  rw [evsi_eq_zero_iff u p he.2]
  constructor
  · rintro ⟨a, ha⟩
    exact ⟨a, fun z _ a' => ha z a'⟩
  · rintro ⟨a, ha⟩
    refine ⟨a, fun z a' => ?_⟩
    by_cases hz : 0 < signalProbability p e z
    · exact ha z hz a'
    · have hnn : ∀ θ ∈ Finset.univ, 0 ≤ p θ * e θ z := fun θ _ =>
        mul_nonneg (hp θ) (he.1 θ z)
      have hs : ∑ θ, p θ * e θ z = 0 :=
        le_antisymm (not_lt.mp hz) (Finset.sum_nonneg hnn)
      have h0 := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp hs
      have hv : ∀ b, signalValue u p e z b = 0 := fun b =>
        Finset.sum_eq_zero fun θ hθ => by rw [h0 θ hθ, zero_mul]
      rw [hv, hv]

end DecisionAnalysisProofs
