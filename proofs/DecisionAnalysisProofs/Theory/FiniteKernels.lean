import DecisionAnalysisProofs.Finite.ValueOfInformation
import FiniteKernelsProofs.Finite.Kernel

/-!
# DecisionAnalysisProofs.Theory.FiniteKernels

The link to `FiniteKernelsProofs` (the Lean model of `FiniteKernels.jl`, on which
`DecisionAnalysis.jl` builds its `Experiment` type). At `K = ℝ` an experiment `Θ → Z → ℝ` is a
`FiniteKernelsProofs.Finite.Kernel Θ Z`, and the constructions of this library are the
sibling's, definitionally:

* `garble_eq_comp`: `garble e g = Kernel.comp e g` (`compose_kernel`).
* `deterministic_eq_ofFun`: `deterministic f = Kernel.ofFun f`; in particular the identity
  experiment is `Kernel.idK`.
* `stochasticKernel_iff`, `normalisedKernel_iff`, `nonnegKernel_iff`: the predicates agree.
* **`valueWithInformation_comp_le`**: T6 stated for the sibling's kernels:
  `value(Kernel.comp e g) ≤ value(e)` for `Kernel.Stochastic g`.
* **`isGarblingOf_iff_exists_comp`**: the garbling relation in the sibling's vocabulary.

Not claimed: anything about the array layout of `FiniteKernel` tables, which the sibling
project treats in its `Layout/` modules.
-/

set_option autoImplicit false
-- One context is declared for the file; not every theorem uses every instance.
set_option linter.unusedSectionVars false

namespace DecisionAnalysisProofs

open FiniteKernelsProofs.Finite

variable {A Θ Z Z' C : Type} [Fintype A] [Nonempty A] [Fintype Θ] [Fintype Z] [Fintype Z']

theorem garble_eq_comp (e : Kernel Θ Z) (g : Kernel Z Z') : garble e g = Kernel.comp e g := rfl

theorem deterministic_eq_ofFun [DecidableEq C] (f : Θ → C) :
    (deterministic f : Θ → C → ℝ) = Kernel.ofFun f := rfl

theorem nonnegKernel_iff (e : Kernel Θ Z) : NonnegKernel e ↔ Kernel.Nonneg e := Iff.rfl

theorem normalisedKernel_iff (e : Kernel Θ Z) : NormalisedKernel e ↔ Kernel.Normalised e :=
  Iff.rfl

theorem stochasticKernel_iff (e : Kernel Θ Z) : StochasticKernel e ↔ Kernel.Stochastic e :=
  Iff.rfl

/-- **T6** for `FiniteKernelsProofs` kernels: composing an experiment with a stochastic kernel
cannot raise the value of deciding after it. -/
theorem valueWithInformation_comp_le (u : A → Θ → ℝ) (p : Θ → ℝ) (e : Kernel Θ Z)
    {g : Kernel Z Z'} (hg : Kernel.Stochastic g) :
    valueWithInformation u p (Kernel.comp e g) ≤ valueWithInformation u p e :=
  valueWithInformation_garble_le u p e hg

theorem isGarblingOf_iff_exists_comp (e' : Kernel Θ Z') (e : Kernel Θ Z) :
    IsGarblingOf e' e ↔ ∃ g : Kernel Z Z', Kernel.Stochastic g ∧ e' = Kernel.comp e g :=
  Iff.rfl

end DecisionAnalysisProofs
