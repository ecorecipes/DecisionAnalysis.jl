import DecisionAnalysisProofs.Finite.ValueOfInformation
import DecisionAnalysisProofs.Finite.Credal

/-!
# DecisionAnalysisProofs.Finite.ImplementationAndModel

Exact finite-model properties for four Track 3 additions in `DecisionAnalysis.jl`:

* **`evim`** (`src/voi.jl`, Fenwick, Claxton & Sculpher 2008): expected value of perfect
  implementation (`evpim`), specific implementation (`evsim`), and the additive decomposition
  of information plus perfect implementation (`EVSI + EVPIM`).
* **`evmi`** (`src/voi.jl`): expected value of model improvement across a weighted ensemble of
  rival models (`evmiEnsemble`) and for replacing an approximate model by truth (`evmiPair`).
* **`prune_generators`** (`src/programs.jl`, `generator_redundancy_model`): dropping any
  generating point that lies in the convex hull of the remaining generators leaves the credal
  convex hull, `lowerExpectedUtility`, `upperExpectedUtility`, `IsGammaMaximin`,
  `IsEAdmissible`, `IsMaximal`, and `IsIntervalUndominated` strictly unchanged.
* **`constrained_policy_model`** (`src/programs.jl`): weak duality bound for mixed policies
  subject to simplex, expectation (`≤`, `≥`), and chance constraints.

Not claimed: solver attainment in `constrained_policy_model` or floating-point rounding
tolerances (`atol`, `rounding_tol`).
-/

set_option autoImplicit false
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open Finset

variable {A Θ Z I M C₁ C₂ : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z]
  [Fintype I] [Nonempty I] [Fintype M] [Nonempty M] [Fintype C₁] [Fintype C₂]
  {K : Type} [Field K] [LinearOrder K] [IsStrictOrderedRing K]

/-! ### Expected value of implementation (`evim`) -/

/-- Expected value of perfect implementation: `V₀ - ∑_a π a · E[u a]`. -/
def evpim (u : A → Θ → K) (p : Θ → K) (π : A → K) : K :=
  actNowValue u p - ∑ a, π a * expectedUtility u p a

/-- Expected value of specific implementation moving from adoption `π` to `π'`. -/
def evsim (u : A → Θ → K) (p : Θ → K) (π π' : A → K) : K :=
  ∑ a, (π' a - π a) * expectedUtility u p a

/-- Perfect implementation cannot lose expected utility (`EVPIM ≥ 0`). -/
theorem evpim_nonneg (u : A → Θ → K) (p : Θ → K) {π : A → K}
    (hπ0 : ∀ a, 0 ≤ π a) (hπ1 : ∑ a, π a = 1) : 0 ≤ evpim u p π := by
  unfold evpim
  have hle : ∑ a, π a * expectedUtility u p a ≤ actNowValue u p :=
    calc ∑ a, π a * expectedUtility u p a
        ≤ ∑ a, π a * actNowValue u p :=
          Finset.sum_le_sum fun a _ =>
            mul_le_mul_of_nonneg_left (expectedUtility_le_actNowValue u p a) (hπ0 a)
      _ = actNowValue u p := by rw [← Finset.sum_mul, hπ1, one_mul]
  linarith

/-- When current adoption puts positive probability only on Bayes actions, `EVPIM = 0`. -/
theorem evpim_eq_zero_of_bayes_support (u : A → Θ → K) (p : Θ → K) {π : A → K}
    (hπ1 : ∑ a, π a = 1) (hsupp : ∀ a, π a ≠ 0 → IsBayes u p a) :
    evpim u p π = 0 := by
  unfold evpim
  have hsum : ∑ a, π a * expectedUtility u p a = actNowValue u p := by
    have hterm : ∀ a, π a * expectedUtility u p a = π a * actNowValue u p := by
      intro a
      by_cases ha : π a = 0
      · simp [ha]
      · rw [(isBayes_iff u p a).mp (hsupp a ha)]
    rw [Finset.sum_congr rfl fun a _ => hterm a, ← Finset.sum_mul, hπ1, one_mul]
  linarith

/-- Specific implementation gains at most as much as perfect implementation (`EVSIM ≤ EVPIM`). -/
theorem evsim_le_evpim (u : A → Θ → K) (p : Θ → K) (π : A → K) {π' : A → K}
    (hπ'0 : ∀ a, 0 ≤ π' a) (hπ'1 : ∑ a, π' a = 1) :
    evsim u p π π' ≤ evpim u p π := by
  unfold evsim evpim
  have hsplit : ∑ a, (π' a - π a) * expectedUtility u p a =
      ∑ a, π' a * expectedUtility u p a - ∑ a, π a * expectedUtility u p a := by
    simp [sub_mul, Finset.sum_sub_distrib]
  rw [hsplit]
  have hle : ∑ a, π' a * expectedUtility u p a ≤ actNowValue u p :=
    calc ∑ a, π' a * expectedUtility u p a
        ≤ ∑ a, π' a * actNowValue u p :=
          Finset.sum_le_sum fun a _ =>
            mul_le_mul_of_nonneg_left (expectedUtility_le_actNowValue u p a) (hπ'0 a)
      _ = actNowValue u p := by rw [← Finset.sum_mul, hπ'1, one_mul]
  linarith

/-- Total value of information plus perfect implementation equals `EVPIM + EVSI`
(backing `evim(problem, experiment; implementation)`). -/
theorem evim_with_experiment_eq_add (u : A → Θ → K) (p : Θ → K) (e : Θ → Z → K) (π : A → K) :
    valueWithInformation u p e - ∑ a, π a * expectedUtility u p a =
      evpim u p π + evsi u p e := by
  unfold evpim evsi
  ring

/-! ### Expected value of model improvement (`evmi`) -/

/-- Ensemble EVMI across rival models `m ∈ M` with weights `w m` and action expectations
`E m a` (`evmi(problems, weights)`). -/
def evmiEnsemble (w : M → K) (E : M → A → K) : K :=
  (∑ m, w m * maxOver (E m)) - maxOver fun a => ∑ m, w m * E m a

/-- Ensemble EVMI is always nonnegative for nonnegative model weights. -/
theorem evmiEnsemble_nonneg {w : M → K} (hw : ∀ m, 0 ≤ w m) (E : M → A → K) :
    0 ≤ evmiEnsemble w E := by
  unfold evmiEnsemble
  have hle : maxOver (fun a => ∑ m, w m * E m a) ≤ ∑ m, w m * maxOver (E m) :=
    maxOver_le fun a => Finset.sum_le_sum fun m _ =>
      mul_le_mul_of_nonneg_left (le_maxOver (E m) a) (hw m)
  linarith

/-- Ensemble EVMI is zero iff a single action is simultaneously optimal under every
positively-weighted rival model. -/
theorem evmiEnsemble_eq_zero_iff {w : M → K} (hw : ∀ m, 0 ≤ w m) (E : M → A → K) :
    evmiEnsemble w E = 0 ↔ ∃ a, ∀ m, 0 < w m → ∀ a', E m a' ≤ E m a := by
  constructor
  · intro h0
    obtain ⟨a, ha⟩ := exists_eq_maxOver (fun a => ∑ m, w m * E m a)
    refine ⟨a, fun m hm a' => ?_⟩
    have hsum : ∑ m', w m' * (maxOver (E m') - E m' a) = 0 := by
      unfold evmiEnsemble at h0
      simp only [mul_sub, Finset.sum_sub_distrib]
      linarith
    have hnn : ∀ m' ∈ Finset.univ, 0 ≤ w m' * (maxOver (E m') - E m' a) := fun m' _ =>
      mul_nonneg (hw m') (sub_nonneg.mpr (le_maxOver (E m') a))
    have hm0 := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp hsum m (Finset.mem_univ m)
    have hdiff : maxOver (E m) - E m a = 0 := (mul_eq_zero.mp hm0).resolve_left hm.ne'
    have hmax : E m a = maxOver (E m) := by linarith
    rw [hmax]
    exact le_maxOver (E m) a'
  · rintro ⟨a, ha⟩
    have hge : ∑ m, w m * maxOver (E m) ≤ ∑ m, w m * E m a := by
      refine Finset.sum_le_sum fun m _ => ?_
      rcases eq_or_lt_of_le (hw m) with h0 | hpos
      · rw [← h0, zero_mul, zero_mul]
      · have heq : maxOver (E m) = E m a := maxOver_eq_of a rfl (ha m hpos)
        rw [heq]
    have hle : ∑ m, w m * E m a ≤ maxOver (fun a => ∑ m, w m * E m a) :=
      le_maxOver (fun a => ∑ m, w m * E m a) a
    have hnn := evmiEnsemble_nonneg hw E
    unfold evmiEnsemble at hnn ⊢
    linarith

/-- Pairwise EVMI (`evmi(approximate, truth)`) is nonnegative and vanishes iff the approximate
choice is already Bayes-optimal under `truth`. -/
theorem evmiPair_nonneg (euTruth : A → K) (aApprox : A) :
    0 ≤ maxOver euTruth - euTruth aApprox :=
  sub_nonneg.mpr (le_maxOver euTruth aApprox)

theorem evmiPair_eq_zero_iff (euTruth : A → K) (aApprox : A) :
    maxOver euTruth - euTruth aApprox = 0 ↔ ∀ a', euTruth a' ≤ euTruth aApprox := by
  constructor
  · intro h a'
    have heq : euTruth aApprox = maxOver euTruth := by linarith
    rw [heq]
    exact le_maxOver euTruth a'
  · intro h
    rw [maxOver_eq_of aApprox rfl h, sub_self]

/-! ### Generator pruning invariance (`prune_generators`, `generator_redundancy_model`) -/

/-- Generator `k` of `P : I → Θ → K` is redundant when it equals a convex combination of the
remaining generators `{ P i | i ≠ k }` (a feasible point of `generator_redundancy_model`). -/
def IsRedundantGenerator [DecidableEq I] (P : I → Θ → K) (k : I) : Prop :=
  ∃ μ : { i : I // i ≠ k } → K, (∀ i, 0 ≤ μ i) ∧ ∑ i, μ i = 1 ∧
    ∀ θ, ∑ i, μ i * P i.1 θ = P k θ

/-- Dropping a redundant generator preserves every convex combination (the credal hull is
unchanged). -/
theorem prune_redundant_preserves_hull [DecidableEq I] {P : I → Θ → K} {k : I}
    (hred : IsRedundantGenerator P k) (w : I → K) (hw0 : ∀ i, 0 ≤ w i) (hw1 : ∑ i, w i = 1) :
    ∃ w' : { i : I // i ≠ k } → K, (∀ i, 0 ≤ w' i) ∧ ∑ i, w' i = 1 ∧
      mixture w' (fun i => P i.1) = mixture w P := by
  obtain ⟨μ, hμ0, hμ1, hμeq⟩ := hred
  let w' : { i : I // i ≠ k } → K := fun i => w i.1 + w k * μ i
  refine ⟨w', fun i => add_nonneg (hw0 i.1) (mul_nonneg (hw0 k) (hμ0 i)), ?_, ?_⟩
  · have hsplit : ∑ i : I, w i = w k + ∑ i : { i : I // i ≠ k }, w i.1 := by
      rw [← Finset.add_sum_erase Finset.univ w (Finset.mem_univ k)]
      congr 1
      symm
      refine Finset.sum_nbij (fun i : { i : I // i ≠ k } => i.1) ?_ ?_ ?_ ?_
      · intro i _; exact Finset.mem_erase.mpr ⟨i.2, Finset.mem_univ i.1⟩
      · intro a _ b _ hab; exact Subtype.ext hab
      · intro i hi; exact ⟨⟨i, (Finset.mem_erase.mp hi).1⟩, Finset.mem_univ _, rfl⟩
      · intro _ _; rfl
    have hsum : ∑ i : { i : I // i ≠ k }, w' i =
        (∑ i : { i : I // i ≠ k }, w i.1) + w k * ∑ i, μ i := by
      simp only [w', Finset.sum_add_distrib, Finset.mul_sum]
    rw [hsum, hμ1, mul_one]
    linarith
  · funext θ
    unfold mixture
    have hsplit : ∑ i : I, w i * P i θ =
        w k * P k θ + ∑ i : { i : I // i ≠ k }, w i.1 * P i.1 θ := by
      rw [← Finset.add_sum_erase Finset.univ (fun i => w i * P i θ) (Finset.mem_univ k)]
      congr 1
      symm
      refine Finset.sum_nbij (fun i : { i : I // i ≠ k } => i.1) ?_ ?_ ?_ ?_
      · intro i _; exact Finset.mem_erase.mpr ⟨i.2, Finset.mem_univ i.1⟩
      · intro a _ b _ hab; exact Subtype.ext hab
      · intro i hi; exact ⟨⟨i, (Finset.mem_erase.mp hi).1⟩, Finset.mem_univ _, rfl⟩
      · intro _ _; rfl
    rw [hsplit, ← hμeq θ, Finset.mul_sum]
    simp only [w', add_mul, Finset.sum_add_distrib]
    have hcomm : ∑ i : { i : I // i ≠ k }, w k * μ i * P i.1 θ =
        ∑ i : { i : I // i ≠ k }, w k * (μ i * P i.1 θ) :=
      Finset.sum_congr rfl fun i _ => mul_assoc _ _ _
    rw [hcomm, add_comm]

/-- Dropping a redundant generator leaves `lowerExpectedUtility` strictly unchanged. -/
theorem prune_redundant_lowerExpectedUtility [DecidableEq I] {P : I → Θ → K} {k : I}
    (hred : IsRedundantGenerator P k) [Nonempty { i : I // i ≠ k }]
    (u : A → Θ → K) (a : A) :
    lowerExpectedUtility u (fun i : { i : I // i ≠ k } => P i.1) a =
      lowerExpectedUtility u P a := by
  obtain ⟨μ, hμ0, hμ1, hμeq⟩ := hred
  let P' : { i : I // i ≠ k } → Θ → K := fun j => P j.1
  apply le_antisymm
  · refine le_minOver fun i => ?_
    by_cases hik : i = k
    · rw [hik]
      have hPk : P k = mixture μ P' := funext fun θ => (hμeq θ).symm
      rw [hPk, expectedUtility_mixture]
      calc lowerExpectedUtility u P' a
          = ∑ j : { j : I // j ≠ k }, μ j * lowerExpectedUtility u P' a := by
            rw [← Finset.sum_mul, hμ1, one_mul]
        _ ≤ ∑ j : { j : I // j ≠ k }, μ j * expectedUtility u (P' j) a :=
            Finset.sum_le_sum fun j _ =>
              mul_le_mul_of_nonneg_left (lowerExpectedUtility_le u P' a j) (hμ0 j)
    · exact lowerExpectedUtility_le u P' a ⟨i, hik⟩
  · exact le_minOver fun i => lowerExpectedUtility_le u P a i.1

/-- Dropping a redundant generator leaves `upperExpectedUtility` strictly unchanged. -/
theorem prune_redundant_upperExpectedUtility [DecidableEq I] {P : I → Θ → K} {k : I}
    (hred : IsRedundantGenerator P k) [Nonempty { i : I // i ≠ k }]
    (u : A → Θ → K) (a : A) :
    upperExpectedUtility u (fun i : { i : I // i ≠ k } => P i.1) a =
      upperExpectedUtility u P a := by
  obtain ⟨μ, hμ0, hμ1, hμeq⟩ := hred
  let P' : { i : I // i ≠ k } → Θ → K := fun j => P j.1
  apply le_antisymm
  · exact maxOver_le fun i => le_upperExpectedUtility u P a i.1
  · refine maxOver_le fun i => ?_
    by_cases hik : i = k
    · rw [hik]
      have hPk : P k = mixture μ P' := funext fun θ => (hμeq θ).symm
      rw [hPk, expectedUtility_mixture]
      calc ∑ j : { j : I // j ≠ k }, μ j * expectedUtility u (P' j) a
          ≤ ∑ j : { j : I // j ≠ k }, μ j * upperExpectedUtility u P' a :=
            Finset.sum_le_sum fun j _ =>
              mul_le_mul_of_nonneg_left (le_upperExpectedUtility u P' a j) (hμ0 j)
        _ = upperExpectedUtility u P' a := by
            rw [← Finset.sum_mul, hμ1, one_mul]
    · exact le_upperExpectedUtility u P' a ⟨i, hik⟩

/-- Dropping a redundant generator preserves `IsEAdmissible`. -/
theorem prune_redundant_isEAdmissible [DecidableEq I] {P : I → Θ → K} {k : I}
    (hred : IsRedundantGenerator P k) [Nonempty { i : I // i ≠ k }]
    (u : A → Θ → K) (a : A) :
    IsEAdmissible u (fun i : { i : I // i ≠ k } => P i.1) a ↔ IsEAdmissible u P a := by
  constructor
  · rintro ⟨w', hw'0, hw'1, hB⟩
    let w : I → K := fun i => if h : i = k then 0 else w' ⟨i, h⟩
    refine ⟨w, fun i => by by_cases h : i = k <;> simp [w, h, hw'0], ?_, ?_⟩
    · have hsum : ∑ i : I, w i = ∑ i : { i : I // i ≠ k }, w' i := by
        rw [← Finset.add_sum_erase Finset.univ w (Finset.mem_univ k)]
        simp only [w, dite_true, zero_add]
        symm
        refine Finset.sum_nbij (fun i : { i : I // i ≠ k } => i.1) ?_ ?_ ?_ ?_
        · intro i _; exact Finset.mem_erase.mpr ⟨i.2, Finset.mem_univ i.1⟩
        · intro x _ y _ hxy; exact Subtype.ext hxy
        · intro i hi; exact ⟨⟨i, (Finset.mem_erase.mp hi).1⟩, Finset.mem_univ _, rfl⟩
        · intro i _; simp [i.2]
      rw [hsum, hw'1]
    · have hmix : mixture w P = mixture w' (fun i : { i : I // i ≠ k } => P i.1) := by
        funext θ
        unfold mixture
        rw [← Finset.add_sum_erase Finset.univ (fun i => w i * P i θ) (Finset.mem_univ k)]
        simp only [w, dite_true, zero_mul, zero_add]
        symm
        refine Finset.sum_nbij (fun i : { i : I // i ≠ k } => i.1) ?_ ?_ ?_ ?_
        · intro i _; exact Finset.mem_erase.mpr ⟨i.2, Finset.mem_univ i.1⟩
        · intro x _ y _ hxy; exact Subtype.ext hxy
        · intro i hi; exact ⟨⟨i, (Finset.mem_erase.mp hi).1⟩, Finset.mem_univ _, rfl⟩
        · intro i _; simp [i.2]
      rwa [hmix]
  · rintro ⟨w, hw0, hw1, hB⟩
    obtain ⟨w', hw'0, hw'1, hmix⟩ := prune_redundant_preserves_hull hred w hw0 hw1
    refine ⟨w', hw'0, hw'1, ?_⟩
    rwa [hmix]

/-- Dropping a redundant generator preserves `IsMaximal`. -/
theorem prune_redundant_isMaximal [DecidableEq I] {P : I → Θ → K} {k : I}
    (hred : IsRedundantGenerator P k) [Nonempty { i : I // i ≠ k }]
    (u : A → Θ → K) (a : A) :
    IsMaximal u (fun i : { i : I // i ≠ k } => P i.1) a ↔ IsMaximal u P a := by
  obtain ⟨μ, hμ0, hμ1, hμeq⟩ := hred
  constructor
  · intro h ⟨a', hdom⟩
    exact h ⟨a', fun i => hdom i.1⟩
  · intro h ⟨a', hdom⟩
    refine h ⟨a', fun i => ?_⟩
    by_cases hik : i = k
    · rw [hik]
      have hPk : P k = mixture μ (fun j : { j : I // j ≠ k } => P j.1) :=
        funext fun θ => (hμeq θ).symm
      rw [hPk]
      exact expectedUtility_mixture_lt hdom hμ0 hμ1
    · exact hdom ⟨i, hik⟩

/-! ### Weak duality for `constrained_policy_model` -/

/-- A mixed policy `μ` feasible for `constrained_policy_model`: simplex plus `≤`-expectation
constraints `cLe` and `≥`-expectation/chance constraints `cGe`. -/
def ConstrainedPolicyFeasible (cLe : C₁ → A → K) (bLe : C₁ → K)
    (cGe : C₂ → A → K) (bGe : C₂ → K) (μ : A → K) : Prop :=
  (∀ a, 0 ≤ μ a) ∧ ∑ a, μ a = 1 ∧
    (∀ i, ∑ a, cLe i a * μ a ≤ bLe i) ∧
    (∀ j, bGe j ≤ ∑ a, cGe j a * μ a)

/-- Dual feasibility for `constrained_policy_model`: scalar `ν` for the simplex constraint,
nonnegative `yLe` for `≤`-constraints, and nonnegative `yGe` for `≥`-constraints. -/
def ConstrainedDualFeasible (eu : A → K) (cLe : C₁ → A → K) (cGe : C₂ → A → K)
    (ν : K) (yLe : C₁ → K) (yGe : C₂ → K) : Prop :=
  (∀ i, 0 ≤ yLe i) ∧ (∀ j, 0 ≤ yGe j) ∧
    ∀ a, eu a ≤ ν + (∑ i, yLe i * cLe i a) - ∑ j, yGe j * cGe j a

/-- **Weak duality for `constrained_policy_model`**: any dual-feasible `(ν, yLe, yGe)` gives a
rigorous upper bound on the expected utility of every feasible mixed policy `μ`. -/
theorem constrained_policy_le_dual {eu : A → K} {cLe : C₁ → A → K} {bLe : C₁ → K}
    {cGe : C₂ → A → K} {bGe : C₂ → K} {μ : A → K} {ν : K} {yLe : C₁ → K} {yGe : C₂ → K}
    (hP : ConstrainedPolicyFeasible cLe bLe cGe bGe μ)
    (hD : ConstrainedDualFeasible eu cLe cGe ν yLe yGe) :
    ∑ a, eu a * μ a ≤ ν + (∑ i, yLe i * bLe i) - ∑ j, yGe j * bGe j := by
  obtain ⟨hμ0, hμ1, hle, hge⟩ := hP
  obtain ⟨hy1, hy2, hdual⟩ := hD
  have h1 : ∑ a, eu a * μ a ≤
      ∑ a, (ν + (∑ i, yLe i * cLe i a) - ∑ j, yGe j * cGe j a) * μ a :=
    Finset.sum_le_sum fun a _ => mul_le_mul_of_nonneg_right (hdual a) (hμ0 a)
  have hexp : ∑ a, (ν + (∑ i, yLe i * cLe i a) - ∑ j, yGe j * cGe j a) * μ a =
      ν + (∑ i, yLe i * ∑ a, cLe i a * μ a) - ∑ j, yGe j * ∑ a, cGe j a * μ a := by
    simp only [sub_mul, add_mul, Finset.sum_sub_distrib, Finset.sum_add_distrib]
    rw [← Finset.mul_sum, hμ1, mul_one]
    congr 1
    · congr 1
      simp only [Finset.sum_mul, Finset.mul_sum]
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun a _ => by ring
    · simp only [Finset.sum_mul, Finset.mul_sum]
      rw [Finset.sum_comm]
      exact Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun a _ => by ring
  have h2 : ∑ i, yLe i * ∑ a, cLe i a * μ a ≤ ∑ i, yLe i * bLe i :=
    Finset.sum_le_sum fun i _ => mul_le_mul_of_nonneg_left (hle i) (hy1 i)
  have h3 : ∑ j, yGe j * bGe j ≤ ∑ j, yGe j * ∑ a, cGe j a * μ a :=
    Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hge j) (hy2 j)
  linarith

end DecisionAnalysisProofs
