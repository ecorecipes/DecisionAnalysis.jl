import DecisionAnalysisProofs.Finite.Regret
import DecisionAnalysisProofs.Finite.ValueOfInformation

/-!
# DecisionAnalysisProofs.Certificate.Check

The proved checker behind `lake exe check_certificate` for documents written by
`export_decision_certificate` (`src/certificates.jl`, schema `"decision-certificate/1"`).

## What is checked

A `Raw` certificate is the decoded JSON: the payoff table (rows are actions), the prior, an
optional likelihood (rows are signals, columns are states, as in `Experiment`), and the
`Claims`. `Raw.instance?` turns the lists into the functions `u : Fin (a+1) → Fin (s+1) → ℚ`
and `p : Fin (s+1) → ℚ` when the table is rectangular and non-empty. `check` then decides
`Accepts`:

* the prior is nonnegative and sums to one;
* `CoreMatches`: `expected_utility`, `act_now_value`, `bayes_actions`, `regret`,
  `expected_regret`, `max_regret`, `minimax_regret_actions` and `evpi` are **equal** to the
  definitions of `Finite/Problem.lean` at `K = ℚ`, tabulated in table order; the two index
  lists are the zero-based positions, ascending, of all actions attaining the optimum (exact
  ties, as `decide(...; atol=0, rtol=0)` reports them);
* `InfoMatches`: without a likelihood, no information claim is present; with one, the
  likelihood is rectangular and column-stochastic and `value_with_information` and `evsi`
  equal `valueWithInformation` and `evsi`.

## Soundness

* **`check_sound`**: if `check r = true` then the instance exists and the three groups above
  hold.
* **`certificate_spec`**: the consequences, as a `Certified` structure: an index is listed
  in `bayes_actions` iff it is a Bayes action (`IsBayes`) and in `minimax_regret_actions` iff
  it is a minimax-regret action (`IsMinimaxRegret`); the claimed EVPI is nonnegative and is
  the minimum of the claimed expected regrets; with an experiment, `0 ≤ evsi ≤ evpi`.

The checker runs the verified definitions themselves (`Finset.sup'` and finite sums over
`Fin n` in `ℚ`); there is no second implementation to trust.

Not claimed: correctness of the JSON parser and of the printing code in
`CheckCertificate.lean`, which are trusted; anything about the `Float64` path of the Julia
package (a certificate can only be exported from an exact problem); and nothing about action
or state labels beyond their number.
-/

set_option autoImplicit false

namespace DecisionAnalysisProofs.Certificate

open DecisionAnalysisProofs

/-! ### Decidability of the predicates over `ℚ` -/

section Decidable

variable {A Θ Z : Type} [Fintype A] [Fintype Θ] [Fintype Z]

instance (p : Θ → ℚ) : Decidable (NonnegPrior p) := by
  unfold NonnegPrior; infer_instance

instance (p : Θ → ℚ) : Decidable (NormalisedPrior p) := by
  unfold NormalisedPrior; infer_instance

instance (e : Θ → Z → ℚ) : Decidable (StochasticKernel e) := by
  unfold StochasticKernel NonnegKernel NormalisedKernel; infer_instance

instance (u : A → Θ → ℚ) (p : Θ → ℚ) (a : A) : Decidable (IsBayes u p a) := by
  unfold IsBayes; infer_instance

end Decidable

/-! ### From lists to tables -/

/-- `rows` has `n` rows of `m` entries each. -/
def rect (rows : List (List ℚ)) (n m : ℕ) : Bool :=
  rows.length == n && rows.all fun r => r.length == m

/-- A list of rows as a table, rows first (`payoff[a][θ]`). -/
def table (rows : List (List ℚ)) (n m : ℕ) : Fin n → Fin m → ℚ :=
  fun i j => (rows.getD i.val []).getD j.val 0

/-- A list as a vector. -/
def vec (xs : List ℚ) (n : ℕ) : Fin n → ℚ := fun i => xs.getD i.val 0

/-- A likelihood matrix (rows are signals, columns are states) as the experiment
`e θ z = likelihood[z][θ]`; also a garbling matrix `G[z'][z]` as the kernel `g z z'`. -/
def kernelOf (rows : List (List ℚ)) (s z : ℕ) : Fin s → Fin z → ℚ :=
  fun θ j => (rows.getD j.val []).getD θ.val 0

/-- The zero-based positions, ascending, of the indices satisfying `P`. -/
def indicesWhere {n : ℕ} (P : Fin n → Prop) [DecidablePred P] : List ℕ :=
  ((List.finRange n).filter fun i => decide (P i)).map Fin.val

theorem mem_indicesWhere {n : ℕ} (P : Fin n → Prop) [DecidablePred P] (i : Fin n) :
    i.val ∈ indicesWhere P ↔ P i := by
  unfold indicesWhere
  rw [List.mem_map]
  constructor
  · rintro ⟨j, hj, hji⟩
    have hij : j = i := Fin.ext hji
    subst hij
    simpa using hj
  · intro h
    exact ⟨i, by simp [h], rfl⟩

/-! ### Certificates -/

/-- The `claims` object of a certificate. -/
structure Claims where
  /-- `expected_utility`: one entry per action. -/
  expectedUtility : List ℚ
  /-- `act_now_value`. -/
  actNowValue : ℚ
  /-- `bayes_actions`: zero-based indices. -/
  bayesActions : List ℕ
  /-- `regret`: rows are actions. -/
  regret : List (List ℚ)
  /-- `expected_regret`: one entry per action. -/
  expectedRegret : List ℚ
  /-- `max_regret`: one entry per action. -/
  maxRegret : List ℚ
  /-- `minimax_regret_actions`: zero-based indices. -/
  minimaxRegretActions : List ℕ
  /-- `evpi`. -/
  evpi : ℚ
  /-- `value_with_information`, present exactly when there is an experiment. -/
  valueWithInformation : Option ℚ
  /-- `evsi`, present exactly when there is an experiment. -/
  evsi : Option ℚ

/-- A decoded `"decision-certificate/1"` document. -/
structure Raw where
  /-- `payoff`: rows are actions, columns are states. -/
  payoff : List (List ℚ)
  /-- `prior`: one entry per state. -/
  prior : List ℚ
  /-- `likelihood`: rows are signals, columns are states. -/
  likelihood : Option (List (List ℚ))
  /-- `claims`. -/
  claims : Claims

/-- A non-empty decision problem over `ℚ`. -/
structure Instance where
  /-- One less than the number of actions. -/
  a : ℕ
  /-- One less than the number of states. -/
  s : ℕ
  /-- The payoff table. -/
  u : Fin (a + 1) → Fin (s + 1) → ℚ
  /-- The prior. -/
  p : Fin (s + 1) → ℚ

/-- The decision problem of a certificate, when its payoff table is rectangular, non-empty and
as wide as the prior. -/
def Raw.instance? (r : Raw) : Option Instance :=
  match r.payoff.length, r.prior.length with
  | a + 1, s + 1 =>
      if rect r.payoff (a + 1) (s + 1) then
        some ⟨a, s, table r.payoff (a + 1) (s + 1), vec r.prior (s + 1)⟩
      else none
  | _, _ => none

/-- Every claim that needs no experiment equals the verified definition. -/
def CoreMatches {a s : ℕ} (u : Fin (a + 1) → Fin (s + 1) → ℚ) (p : Fin (s + 1) → ℚ)
    (c : Claims) : Prop :=
  c.expectedUtility = List.ofFn (expectedUtility u p) ∧
  c.actNowValue = actNowValue u p ∧
  c.bayesActions = indicesWhere (fun i => expectedUtility u p i = actNowValue u p) ∧
  c.regret = List.ofFn (fun i => List.ofFn (regret u i)) ∧
  c.expectedRegret = List.ofFn (expectedRegret u p) ∧
  c.maxRegret = List.ofFn (maxRegret u) ∧
  c.minimaxRegretActions = indicesWhere (fun i => maxRegret u i = minimaxRegretValue u) ∧
  c.evpi = evpi u p

instance {a s : ℕ} (u : Fin (a + 1) → Fin (s + 1) → ℚ) (p : Fin (s + 1) → ℚ) (c : Claims) :
    Decidable (CoreMatches u p c) := by
  unfold CoreMatches; infer_instance

/-- The information claims equal the verified definitions for the stated likelihood, and are
absent when there is none. -/
def InfoMatches {a s : ℕ} (u : Fin (a + 1) → Fin (s + 1) → ℚ) (p : Fin (s + 1) → ℚ)
    (lik : Option (List (List ℚ))) (c : Claims) : Prop :=
  match lik with
  | none => c.valueWithInformation = none ∧ c.evsi = none
  | some rows =>
      rect rows rows.length (s + 1) = true ∧
      StochasticKernel (kernelOf rows (s + 1) rows.length) ∧
      c.valueWithInformation =
        some (valueWithInformation u p (kernelOf rows (s + 1) rows.length)) ∧
      c.evsi = some (evsi u p (kernelOf rows (s + 1) rows.length))

instance {a s : ℕ} (u : Fin (a + 1) → Fin (s + 1) → ℚ) (p : Fin (s + 1) → ℚ)
    (lik : Option (List (List ℚ))) (c : Claims) : Decidable (InfoMatches u p lik c) := by
  unfold InfoMatches; split <;> infer_instance

/-- What the checker decides. -/
def Accepts (r : Raw) : Prop :=
  match r.instance? with
  | none => False
  | some I =>
      NonnegPrior I.p ∧ NormalisedPrior I.p ∧ CoreMatches I.u I.p r.claims ∧
        InfoMatches I.u I.p r.likelihood r.claims

instance (r : Raw) : Decidable (Accepts r) := by
  unfold Accepts; split <;> infer_instance

/-- The certificate checker run by `lake exe check_certificate`. -/
def check (r : Raw) : Bool := decide (Accepts r)

/-- **Soundness of the checker.** An accepted certificate has a well-formed problem, a
probability vector as prior, and claims equal to the verified definitions. -/
theorem check_sound {r : Raw} (h : check r = true) :
    ∃ I : Instance, r.instance? = some I ∧ NonnegPrior I.p ∧ NormalisedPrior I.p ∧
      CoreMatches I.u I.p r.claims ∧ InfoMatches I.u I.p r.likelihood r.claims := by
  have h' : Accepts r := of_decide_eq_true h
  unfold Accepts at h'
  split at h'
  · exact h'.elim
  · rename_i I hI
    exact ⟨I, hI, h'⟩

/-! ### What an accepted certificate establishes -/

section Consequences

variable {a s : ℕ} {u : Fin (a + 1) → Fin (s + 1) → ℚ} {p : Fin (s + 1) → ℚ} {c : Claims}

/-- An index is listed in `bayes_actions` iff it is a Bayes action. -/
theorem CoreMatches.bayesActions_iff (h : CoreMatches u p c) (i : Fin (a + 1)) :
    i.val ∈ c.bayesActions ↔ IsBayes u p i := by
  rw [h.2.2.1, mem_indicesWhere, isBayes_iff]

/-- An index is listed in `minimax_regret_actions` iff it is a minimax-regret action. -/
theorem CoreMatches.minimaxRegretActions_iff (h : CoreMatches u p c) (i : Fin (a + 1)) :
    i.val ∈ c.minimaxRegretActions ↔ IsMinimaxRegret u i := by
  rw [h.2.2.2.2.2.2.1, mem_indicesWhere, isMinimaxRegret_iff]

/-- The claimed EVPI is nonnegative. -/
theorem CoreMatches.evpi_nonneg (h : CoreMatches u p c) (hp : NonnegPrior p) : 0 ≤ c.evpi := by
  rw [h.2.2.2.2.2.2.2]
  exact DecisionAnalysisProofs.evpi_nonneg u hp

/-- The claimed EVPI is the minimum expected regret. -/
theorem CoreMatches.evpi_eq_minOver_expectedRegret (h : CoreMatches u p c) :
    c.evpi = minOver (expectedRegret u p) := by
  rw [h.2.2.2.2.2.2.2, minOver_expectedRegret_eq_evpi]

/-- With an experiment, the claimed EVSI lies between zero and the claimed EVPI. -/
theorem InfoMatches.evsi_bounds {rows : List (List ℚ)} (hi : InfoMatches u p (some rows) c)
    (hc : CoreMatches u p c) (hp : NonnegPrior p) :
    ∃ v, c.evsi = some v ∧ 0 ≤ v ∧ v ≤ c.evpi := by
  obtain ⟨-, he, -, hv⟩ := hi
  refine ⟨_, hv, evsi_nonneg u p he.2, ?_⟩
  rw [hc.2.2.2.2.2.2.2]
  exact evsi_le_evpi u hp he

end Consequences

/-- The statement certified by an accepted certificate, for its decision problem `I`. -/
structure Certified (I : Instance) (lik : Option (List (List ℚ))) (c : Claims) : Prop where
  /-- The prior is nonnegative. -/
  prior_nonneg : NonnegPrior I.p
  /-- The prior sums to one. -/
  prior_normalised : NormalisedPrior I.p
  /-- The claims without an experiment equal the verified definitions. -/
  core : CoreMatches I.u I.p c
  /-- The information claims equal the verified definitions. -/
  info : InfoMatches I.u I.p lik c
  /-- `bayes_actions` lists exactly the Bayes actions. -/
  bayes_iff : ∀ i : Fin (I.a + 1), i.val ∈ c.bayesActions ↔ IsBayes I.u I.p i
  /-- `minimax_regret_actions` lists exactly the minimax-regret actions. -/
  minimax_iff : ∀ i : Fin (I.a + 1), i.val ∈ c.minimaxRegretActions ↔ IsMinimaxRegret I.u i
  /-- The claimed EVPI is nonnegative. -/
  evpi_nonneg : 0 ≤ c.evpi
  /-- The claimed EVPI is the minimum expected regret. -/
  evpi_eq_min_regret : c.evpi = minOver (expectedRegret I.u I.p)
  /-- With an experiment, `0 ≤ evsi ≤ evpi`. -/
  evsi_bounds : ∀ rows, lik = some rows → ∃ v, c.evsi = some v ∧ 0 ≤ v ∧ v ≤ c.evpi

/-- **Certificate specification.** If `check_certificate` accepts, the certificate's problem
is well formed and every claim is the verified quantity, with the listed consequences. -/
theorem certificate_spec {r : Raw} (h : check r = true) :
    ∃ I : Instance, r.instance? = some I ∧ Certified I r.likelihood r.claims := by
  obtain ⟨I, hI, hp, hn, hc, hi⟩ := check_sound h
  refine ⟨I, hI, hp, hn, hc, hi, hc.bayesActions_iff, hc.minimaxRegretActions_iff,
    hc.evpi_nonneg hp, hc.evpi_eq_minOver_expectedRegret, ?_⟩
  intro rows hrows
  rw [hrows] at hi
  exact hi.evsi_bounds hc hp

end DecisionAnalysisProofs.Certificate
