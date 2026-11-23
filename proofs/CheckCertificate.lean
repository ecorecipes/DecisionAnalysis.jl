/-
`lake exe check_certificate FILE.json`

Reads one JSON document, parses it with `Lean.Json.parse`, and dispatches on its `"schema"`:

* `"decision-certificate/1"` (written by DecisionAnalysis.jl's `export_decision_certificate`,
  `src/certificates.jl`): decoded to `Certificate.Raw` and decided by `Certificate.check`
  (`DecisionAnalysisProofs/Certificate/Check.lean`). Theorem `Certificate.certificate_spec`
  applies to every accepted document: each claim equals the verified definition over ℚ.
* `"garbling-witness/1"`: `{"experiment", "garbled", "garbling"}`, the two likelihoods (rows
  are signals, columns are states) and the matrix `G[z'][z] = P(z' | z)` returned by
  `is_garbling_of`; decided by `Certificate.checkGarbling`
  (`DecisionAnalysisProofs/Certificate/Witness.lean`, theorem `RawGarbling.accepts_sound`).
* `"e-admissible-witness/1"`: `{"payoff", "priors", "weights", "prior", "action"}`, the payoff
  table (rows are actions), the extreme points of the credal set (one per row), the convex
  weights, the witness prior and the zero-based action index; decided by
  `Certificate.checkEAdmissible` (theorem `RawEAdmissible.accepts_sound`).

Every number is a JSON string `"n/d"` or `"n"` (a JSON integer is also accepted); indices are
JSON integers counted from zero. The exit code is 0 exactly when the document is well formed
and the proved checker accepts it; 1 when it is rejected; 2 when the file cannot be read or
decoded (wrong schema, ragged matrix, label counts that do not match, a number that is not a
rational).

The parse, the decoding into `Raw*` structures and the printing below are trusted, not proved.
The verdict is the value of the proved checker, and the per-claim lines are `decide`s of the
conjuncts of `Certificate.CoreMatches` and `Certificate.InfoMatches`.
-/
import DecisionAnalysisProofs.Certificate.Witness
import Lean.Data.Json.Parser

open Lean DecisionAnalysisProofs DecisionAnalysisProofs.Certificate

def verdict (b : Bool) : String := if b then "ok" else "FAIL"

/-- `"n/d"` or `"n"` as a rational. -/
def parseRat (s : String) : Except String ℚ :=
  match s.splitOn "/" with
  | [n] =>
    match n.toInt? with
    | some i => pure (i : ℚ)
    | none => throw s!"not a rational: {s.quote}"
  | [n, d] =>
    match n.toInt?, d.toNat? with
    | some i, some k =>
      if k = 0 then throw s!"zero denominator: {s.quote}" else pure ((i : ℚ) / (k : ℚ))
    | _, _ => throw s!"not a rational: {s.quote}"
  | _ => throw s!"not a rational: {s.quote}"

def ratOfJson : Json → Except String ℚ
  | .str s => parseRat s
  | .num n => if n.exponent = 0 then pure (n.mantissa : ℚ) else throw "numbers must be exact"
  | _ => throw "expected a rational as a string"

def ratList (j : Json) : Except String (List ℚ) := do
  let arr ← j.getArr?
  arr.toList.mapM ratOfJson

def ratMatrix (j : Json) : Except String (List (List ℚ)) := do
  let arr ← j.getArr?
  arr.toList.mapM ratList

def natList (j : Json) : Except String (List ℕ) := do
  let arr ← j.getArr?
  arr.toList.mapM fun x => x.getNat?

def field (j : Json) (k : String) : Except String Json :=
  match j.getObjVal? k with
  | .ok v => pure v
  | .error _ => throw s!"missing field {k.quote}"

def optField (j : Json) (k : String) : Option Json :=
  match j.getObjVal? k with
  | .ok v => some v
  | .error _ => none

def labelCount (j : Json) (k : String) : Except String ℕ := do
  let arr ← (← field j k).getArr?
  for x in arr do
    let _ ← x.getStr?
  pure arr.size

def requireRect (what : String) (rows : List (List ℚ)) (n m : ℕ) : Except String Unit :=
  if rect rows n m then pure () else throw s!"{what} is not a {n} by {m} matrix"

def decodeCertificate (j : Json) : Except String Raw := do
  let nA ← labelCount j "actions"
  let nS ← labelCount j "states"
  let payoff ← ratMatrix (← field j "payoff")
  let prior ← ratList (← field j "prior")
  if nA = 0 ∨ nS = 0 then throw "a certificate needs at least one action and one state"
  requireRect "payoff" payoff nA nS
  if prior.length ≠ nS then throw s!"prior has {prior.length} entries for {nS} states"
  let likelihood ← match optField j "likelihood" with
    | none =>
      if (optField j "signals").isSome then throw "signals without likelihood" else pure none
    | some l => do
      let nZ ← labelCount j "signals"
      let rows ← ratMatrix l
      requireRect "likelihood" rows nZ nS
      pure (some rows)
  let c ← field j "claims"
  let optRat (k : String) : Except String (Option ℚ) :=
    match optField c k with
    | none => pure none
    | some v => do pure (some (← ratOfJson v))
  pure
    { payoff, prior, likelihood,
      claims :=
        { expectedUtility := ← ratList (← field c "expected_utility")
          actNowValue := ← ratOfJson (← field c "act_now_value")
          bayesActions := ← natList (← field c "bayes_actions")
          regret := ← ratMatrix (← field c "regret")
          expectedRegret := ← ratList (← field c "expected_regret")
          maxRegret := ← ratList (← field c "max_regret")
          minimaxRegretActions := ← natList (← field c "minimax_regret_actions")
          evpi := ← ratOfJson (← field c "evpi")
          valueWithInformation := ← optRat "value_with_information"
          evsi := ← optRat "evsi" } }

def decodeGarbling (j : Json) : Except String RawGarbling := do
  let experiment ← ratMatrix (← field j "experiment")
  let garbled ← ratMatrix (← field j "garbled")
  let garbling ← ratMatrix (← field j "garbling")
  pure { nStates := (experiment.headD []).length, experiment, garbled, garbling }

def decodeEAdmissible (j : Json) : Except String RawEAdmissible := do
  pure
    { payoff := ← ratMatrix (← field j "payoff")
      priors := ← ratMatrix (← field j "priors")
      weights := ← ratList (← field j "weights")
      prior := ← ratList (← field j "prior")
      action := ← (← field j "action").getNat? }

/-- The conjuncts of `CoreMatches` and `InfoMatches`, decided one by one (printing only). -/
def components (I : Instance) (r : Raw) : List (String × Bool) :=
  let c := r.claims
  [("prior nonnegative", decide (NonnegPrior I.p)),
    ("prior sums to one", decide (NormalisedPrior I.p)),
    ("expected_utility", decide (c.expectedUtility = List.ofFn (expectedUtility I.u I.p))),
    ("act_now_value", decide (c.actNowValue = actNowValue I.u I.p)),
    ("bayes_actions", decide (c.bayesActions =
      indicesWhere (fun i => expectedUtility I.u I.p i = actNowValue I.u I.p))),
    ("regret", decide (c.regret = List.ofFn (fun i => List.ofFn (regret I.u i)))),
    ("expected_regret", decide (c.expectedRegret = List.ofFn (expectedRegret I.u I.p))),
    ("max_regret", decide (c.maxRegret = List.ofFn (maxRegret I.u))),
    ("minimax_regret_actions", decide (c.minimaxRegretActions =
      indicesWhere (fun i => maxRegret I.u i = minimaxRegretValue I.u))),
    ("evpi", decide (c.evpi = evpi I.u I.p))] ++
  match r.likelihood with
  | none =>
    [("no information claims without an experiment",
      decide (c.valueWithInformation = none ∧ c.evsi = none))]
  | some rows =>
    let e := kernelOf rows (I.s + 1) rows.length
    [("likelihood column-stochastic", decide (StochasticKernel e)),
      ("value_with_information",
        decide (c.valueWithInformation = some (valueWithInformation I.u I.p e))),
      ("evsi", decide (c.evsi = some (evsi I.u I.p e)))]

def runCertificate (j : Json) : IO UInt32 := do
  match decodeCertificate j with
  | .error e => IO.eprintln s!"malformed certificate: {e}"; return 2
  | .ok r =>
    match r.instance? with
    | none => IO.eprintln "malformed certificate: empty or ragged payoff table"; return 2
    | some I =>
      IO.println s!"schema: decision-certificate/1"
      IO.println s!"actions: {I.a + 1}, states: {I.s + 1}"
      for (name, ok) in components I r do
        IO.println s!"{name}: {verdict ok}"
      let ok := check r
      IO.println s!"accepted: {if ok then "yes" else "no"}"
      if ok then
        IO.println "theorem Certificate.certificate_spec: applies"
        return 0
      else
        return 1

def runGarbling (j : Json) : IO UInt32 := do
  match decodeGarbling j with
  | .error e => IO.eprintln s!"malformed garbling witness: {e}"; return 2
  | .ok r =>
    IO.println s!"schema: garbling-witness/1"
    IO.println s!"states: {r.nStates}, signals: {r.experiment.length}, garbled signals: {r.garbled.length}"
    let shapes := rect r.experiment r.experiment.length r.nStates &&
      rect r.garbled r.garbled.length r.nStates &&
      rect r.garbling r.garbled.length r.experiment.length
    IO.println s!"shapes: {verdict shapes}"
    IO.println s!"experiment column-stochastic: {verdict (decide (StochasticKernel r.e))}"
    IO.println s!"garbling column-stochastic: {verdict (decide (StochasticKernel r.g))}"
    IO.println s!"garbled = garbling * experiment: {verdict (decide (r.e' = garble r.e r.g))}"
    let ok := checkGarbling r
    IO.println s!"accepted: {if ok then "yes" else "no"}"
    if ok then
      IO.println "theorem Certificate.RawGarbling.accepts_sound: applies"
      return 0
    else
      return 1

def runEAdmissible (j : Json) : IO UInt32 := do
  match decodeEAdmissible j with
  | .error e => IO.eprintln s!"malformed E-admissibility witness: {e}"; return 2
  | .ok r =>
    IO.println s!"schema: e-admissible-witness/1"
    IO.println s!"actions: {r.payoff.length}, states: {r.prior.length}, extreme points: {r.weights.length}, action: {r.action}"
    let ok := checkEAdmissible r
    IO.println s!"accepted: {if ok then "yes" else "no"}"
    if ok then
      IO.println "theorem Certificate.RawEAdmissible.accepts_sound: applies"
      return 0
    else
      return 1

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] =>
    match ← (IO.FS.readFile path).toBaseIO with
    | .error e => IO.eprintln s!"cannot read {path}: {e}"; return 2
    | .ok text =>
      match Json.parse text with
      | .error e => IO.eprintln s!"not JSON: {e}"; return 2
      | .ok j =>
        match (field j "schema").bind fun s => s.getStr? with
        | .ok "decision-certificate/1" => runCertificate j
        | .ok "garbling-witness/1" => runGarbling j
        | .ok "e-admissible-witness/1" => runEAdmissible j
        | .ok s => IO.eprintln s!"unknown schema {s.quote}"; return 2
        | .error e => IO.eprintln s!"malformed document: {e}"; return 2
  | _ =>
    IO.eprintln "usage: check_certificate FILE.json"
    return 2
