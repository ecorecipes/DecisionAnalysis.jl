# Exact arithmetic and certificates
Simon Frost

- [Motivation](#motivation)
- [The problem](#the-problem)
- [Exact in, exact out](#exact-in-exact-out)
  - [Ties are exact](#ties-are-exact)
  - [Identities hold exactly](#identities-hold-exactly)
- [A decision certificate](#a-decision-certificate)
- [A garbling witness](#a-garbling-witness)
- [An E-admissibility witness](#an-e-admissibility-witness)
- [The linear programmes behind the
  witnesses](#the-linear-programmes-behind-the-witnesses)
- [Checking the files with Lean](#checking-the-files-with-lean)
  - [A wrong claim is rejected](#a-wrong-claim-is-rejected)
- [What is and is not certified](#what-is-and-is-not-certified)
- [What to take away](#what-to-take-away)
- [The series](#the-series)

## Motivation

A decision analysis ends in comparisons: this action is better than that
one, this survey is worth more than it costs. In floating point a
comparison between two nearly equal numbers can come out either way.
`DecisionAnalysis.jl` avoids the question for finite problems: give it
integers and rationals and every number it returns is an exact rational,
every tie is a true tie, and every linear programme is solved exactly.

Exact results can also be *checked*. The package writes a problem and
the numbers it claims for it as a certificate, a small JSON file, and a
separate program recomputes the claims. The checker in `proofs/` is
written in Lean 4, shares no code with the Julia package, and comes with
machine-checked proofs that what it accepts is correct. This vignette
writes the three kinds of file and checks them.

## The problem

The frog pond with Bd, as in the earlier vignettes: three states, three
actions, and utility equal to uninfected frogs minus `w` times infected
frogs. Infected but living frogs are not counted as population. The
certificate below is for `w = 1`, and a second one is written for
`w = 0`.

**All numbers are illustrative**, not data.

``` julia
using DecisionAnalysis
using Random
include(joinpath("..", "_common.jl"))

actions = [:no_intervention, :antifungal_baths, :disinfect_pond]
states = [:absent, :low_prevalence, :high_prevalence]
uninfected = PayoffTable(actions, states, [500 300 100; 470 380 250; 350 340 320])
infected = PayoffTable(actions, states, [0 150 250; 0 60 120; 0 10 20])
prior = FiniteBelief(states, [1 // 2, 3 // 10, 1 // 5])

function bd_problem(w)
    return DecisionProblem(infection_payoff(; uninfected, infected, weight=w), prior;
                           baseline=:no_intervention)
end

problem = bd_problem(1)
swab = Experiment([19//20 2//5 1//20; 1//20 3//5 19//20];
                  signals=[:negative, :positive], states=states)
problem.payoff
```

**PayoffTable{Int64}: 3 actions × 3 states**

| action             | `absent` | `low_prevalence` | `high_prevalence` |
|:-------------------|---------:|-----------------:|------------------:|
| `no_intervention`  |      500 |              150 |              -150 |
| `antifungal_baths` |      470 |              320 |               130 |
| `disinfect_pond`   |      350 |              330 |               300 |

## Exact in, exact out

The payoffs are integers and the probabilities rationals, so the results
are rationals.

``` julia
md_table(["quantity", "exact value", "as a decimal"],
         [[name, exact(x), Float64(x)]
          for (name, x) in ["expected utility of the best action" => act_now_value(problem),
                            "its expected gain over no intervention" =>
                                maximum(expected_gain_over_baseline(problem)),
                            "EVPI" => evpi(problem),
                            "value with the swab" =>
                                value_with_information(problem, swab),
                            "EVSI of the swab" => evsi(problem, swab)]]; digits=4)
```

| quantity                               | exact value | as a decimal |
|:---------------------------------------|------------:|-------------:|
| expected utility of the best action    |         357 |          357 |
| its expected gain over no intervention |          92 |           92 |
| EVPI                                   |          52 |           52 |
| value with the swab                    |     3881/10 |        388.1 |
| EVSI of the swab                       |      311/10 |         31.1 |

``` julia
typeof(evsi(problem, swab))
```

    Rational{Int64}

The same problem in floating point gives floating-point answers that
agree to rounding error, which is what one wants for large problems. The
exact form is for the cases where the comparison itself is the result.

``` julia
float_problem = DecisionProblem(PayoffTable(actions, states,
                                            Float64.(problem.payoff.Q)),
                                Float64.(probabilities(prior));
                                baseline=:no_intervention)
float_swab = Experiment(Float64.(swab.likelihood); states=states)
evsi(float_problem, float_swab), Float64(evsi(problem, swab))
```

    (31.099999999999966, 31.1)

### Ties are exact

At what weight `w` do the baths and disinfection have the same expected
utility? The expected counts are 399 uninfected and 42 infected for the
baths, 341 and 7 for disinfection, so the two are level when 399 − 42w =
341 − 7w, at `w` = 58/35.

``` julia
tie = decide(bd_problem(58 // 35), MaxExpectedUtility())
tie.chosen, first_choice(tie), exact.(tie.scores)
```

    ([:antifungal_baths, :disinfect_pond], :antifungal_baths, ["1418/7", "1647/5", "1647/5"])

Both actions are returned, and the first choice is the one that comes
first in the table. In floating point the same weight does not give a
tie unless a tolerance is allowed for, which `decide` does by default
for floating-point problems (`rtol`).

``` julia
w = 58 / 35
float_tie = DecisionProblem(infection_payoff(; uninfected, infected, weight=w),
                            Float64.(probabilities(prior)))
decide(float_tie, MaxExpectedUtility(); rtol=0).chosen,
decide(float_tie, MaxExpectedUtility()).chosen
```

    ([:antifungal_baths], [:antifungal_baths, :disinfect_pond])

### Identities hold exactly

The identities of the earlier vignettes can be tested with `==` on
random rational problems. The package provides generators for that.

``` julia
rng = Xoshiro(2026)
checks = map(1:200) do _
    p = DecisionProblem(random_payoff_table(rng, 4, 5; T=Rational{Int}),
                        random_belief(rng, 5; T=Rational{Int}))
    e = random_experiment(rng, 3, 5; T=Rational{Int})
    noisy = garble(e, random_experiment(rng, 3, 3; T=Rational{Int}).likelihood)
    return (evpi(p) == minimum(expected_regret(p)),
            0 <= evsi(p, noisy) <= evsi(p, e) <= evpi(p),
            is_garbling_of(noisy, e) !== nothing)
end
md_table(["property, on 200 random rational problems", "holds in all"],
         [["EVPI equals the smallest expected regret", all(c -> c[1], checks)],
          ["0 ≤ EVSI of a garbling ≤ EVSI ≤ EVPI", all(c -> c[2], checks)],
          ["the garbling is recognised as one", all(c -> c[3], checks)]])
```

| property, on 200 random rational problems | holds in all |
|:------------------------------------------|-------------:|
| EVPI equals the smallest expected regret  |          yes |
| 0 ≤ EVSI of a garbling ≤ EVSI ≤ EVPI      |          yes |
| the garbling is recognised as one         |          yes |

## A decision certificate

`export_decision_certificate` writes the problem and the claims as JSON.
Every number is a string `"n/d"` in lowest terms, and action indices
count from zero.

``` julia
out = joinpath(@__DIR__, "_out")
mkpath(out)
certificate = joinpath(out, "bd_w1.json")
export_decision_certificate(certificate, problem; experiment=swab)
print(read(certificate, String))
```

    {
      "schema": "decision-certificate/1",
      "actions": ["no_intervention", "antifungal_baths", "disinfect_pond"],
      "states": ["absent", "low_prevalence", "high_prevalence"],
      "payoff": [["500", "150", "-150"], ["470", "320", "130"], ["350", "330", "300"]],
      "prior": ["1/2", "3/10", "1/5"],
      "signals": ["negative", "positive"],
      "likelihood": [["19/20", "2/5", "1/20"], ["1/20", "3/5", "19/20"]],
      "claims": {
        "expected_utility": ["265", "357", "334"],
        "act_now_value": "357",
        "bayes_actions": [1],
        "regret": [["0", "180", "450"], ["30", "10", "170"], ["150", "0", "0"]],
        "expected_regret": ["144", "52", "75"],
        "max_regret": ["450", "170", "150"],
        "minimax_regret_actions": [2],
        "evpi": "52",
        "value_with_information": "3881/10",
        "evsi": "311/10"
      }
    }

The claims are the expected utility of every action, the value of acting
now, the Bayes actions, the regret table with its expected and maximum
regret, the minimax-regret actions, EVPI, and with an experiment the
value with information and EVSI. The gain over the baseline is not a
separate claim: it is the difference of two entries of
`expected_utility`.

``` julia
export_decision_certificate(joinpath(out, "bd_w0.json"), bd_problem(0); experiment=swab)
CERTIFICATE_SCHEMA
```

    "decision-certificate/1"

A certificate needs exact inputs. A floating-point problem is refused:

``` julia
try
    export_decision_certificate(devnull, float_problem)
catch err
    print(sprint(showerror, err))
end
```

    InexactCertificateError: the payoff has element type Float64; a certificate needs Rational or Integer entries

## A garbling witness

The claim “pooled swabs are a garbling of the swab test” is proved by
exhibiting the noise matrix. `export_garbling_witness` finds it by exact
linear programme, checks it in exact arithmetic (nonnegative, columns
summing to one, product equal to the garbled likelihood) and writes it.

``` julia
pooled = garble(swab, [9//10 1//5; 1//10 4//5]; signals=[:negative, :positive])
garbling = joinpath(out, "pooled_is_garbling_of_swab.json")
export_garbling_witness(garbling, pooled, swab)
print(read(garbling, String))
```

    {
      "schema": "garbling-witness/1",
      "experiment": [["19/20", "2/5", "1/20"], ["1/20", "3/5", "19/20"]],
      "garbled": [["173/200", "12/25", "47/200"], ["27/200", "13/25", "153/200"]],
      "garbling": [["9/10", "1/5"], ["1/10", "4/5"]]
    }

When there is no witness nothing is written:

``` julia
try
    export_garbling_witness(devnull, swab, pooled)
catch err
    print(sprint(showerror, err))
end
```

    WitnessError: the first experiment is not a garbling of the second: there is no witness

A “no” has a proof of its own. If the swab were a garbling of the pooled
swabs it could never be worth more than them, so one decision problem in
which it is worth strictly more settles the matter.
`export_not_garbling_witness` finds such a problem and writes it, as a
payoff table and a prior:

``` julia
not_garbling = joinpath(out, "swab_is_not_garbling_of_pooled.json")
export_not_garbling_witness(not_garbling, swab, pooled)
print(read(not_garbling, String))
```

    {
      "schema": "not-garbling-witness/1",
      "experiment": [["173/200", "12/25", "47/200"], ["27/200", "13/25", "153/200"]],
      "garbled": [["19/20", "2/5", "1/20"], ["1/20", "3/5", "19/20"]],
      "payoff": [["1", "-1", "-1"], ["-1", "1", "1"]],
      "prior": ["1/3", "1/3", "1/3"]
    }

## An E-admissibility witness

The claim “this action is optimal under some prior in the credal set” is
proved by exhibiting the prior and the weights that place it in the set.

``` julia
credal = CredalSet(states, [[1 // 2, 3 // 10, 1 // 5],
                            [7 // 10, 1 // 5, 1 // 10],
                            [3 // 10, 3 // 10, 2 // 5]])
robust = DecisionProblem(problem.payoff, credal; baseline=:no_intervention)
decide(robust, EAdmissible()).chosen
```

    2-element Vector{Symbol}:
     :antifungal_baths
     :disinfect_pond

``` julia
admissible = joinpath(out, "disinfect_is_e_admissible.json")
export_eadmissible_witness(admissible, robust, :disinfect_pond)
print(read(admissible, String))
```

    {
      "schema": "e-admissible-witness/1",
      "payoff": [["500", "150", "-150"], ["470", "320", "130"], ["350", "330", "300"]],
      "priors": [["1/2", "3/10", "1/5"], ["7/10", "1/5", "1/10"], ["3/10", "3/10", "2/5"]],
      "weights": ["35/58", "0", "23/58"],
      "prior": ["61/145", "3/10", "81/290"],
      "action": 2
    }

No intervention is not E-admissible, so it has no witness:

``` julia
try
    export_eadmissible_witness(devnull, robust, :no_intervention)
catch err
    print(sprint(showerror, err))
end
```

    WitnessError: no_intervention is not E-admissible: under every prior in the credal set some other action has a higher expected utility

Here too the “no” can be proved: a mixture of the other actions that is
strictly better than no intervention at every generating point of the
credal set, and therefore under every prior in it.

``` julia
not_admissible = joinpath(out, "no_intervention_is_not_e_admissible.json")
export_not_eadmissible_witness(not_admissible, robust, :no_intervention)
print(read(not_admissible, String))
```

    {
      "schema": "not-e-admissible-witness/1",
      "actions": ["no_intervention", "antifungal_baths", "disinfect_pond"],
      "states": ["absent", "low_prevalence", "high_prevalence"],
      "payoff": [["500", "150", "-150"], ["470", "320", "130"], ["350", "330", "300"]],
      "priors": [["1/2", "3/10", "1/5"], ["7/10", "1/5", "1/10"], ["3/10", "3/10", "2/5"]],
      "action": 0,
      "mixed_action": ["0", "1", "0"]
    }

## The linear programmes behind the witnesses

Each witness is the solution of a linear programme, and the package
gives the programmes themselves: a builder returns a `DecisionProgram`,
which holds a JuMP model with named variables and constraints, and
`solve_program` solves it, exactly when the inputs are rational.

``` julia
programmes = ["`eadmissibility_model(robust, :disinfect_pond)`" =>
                  eadmissibility_model(robust, :disinfect_pond),
              "`mixed_dominance_model(robust, :no_intervention)`" =>
                  mixed_dominance_model(robust, :no_intervention),
              "`evpi_envelope_model(robust)`" => evpi_envelope_model(robust),
              "`garbling_model(pooled, swab)`" => garbling_model(pooled, swab),
              "`separating_payoff_model(swab, pooled)`" => separating_payoff_model(swab, pooled),
              "`deficiency_model(pooled, swab)`" => deficiency_model(pooled, swab)]
md_table(["builder", "programme", "status", "objective", "exact"],
         [begin
              solution = solve_program(program)
              [call, program.name, solution.status, solution.objective, solution.exact]
          end for (call, program) in programmes])
```

| builder | programme | status | objective | exact |
|:---|---:|---:|---:|---:|
| `eadmissibility_model(robust, :disinfect_pond)` | `e_admissibility` | `optimal` | 0 | yes |
| `mixed_dominance_model(robust, :no_intervention)` | `mixed_dominance` | `optimal` | 41 | yes |
| `evpi_envelope_model(robust)` | `evpi_envelope` | `optimal` | 66 | yes |
| `garbling_model(pooled, swab)` | `garbling` | `optimal` | 0 | yes |
| `separating_payoff_model(swab, pooled)` | `separating_payoff` | `optimal` | 0.7 | yes |
| `deficiency_model(pooled, swab)` | `deficiency` | `optimal` | 0.33 | yes |

The dominance programme has a positive optimum because no intervention
is beaten by a margin; the separating programme has one because the swab
is not a garbling of the pooled swabs; the garbling programme is a
feasibility problem and has no objective.

## Checking the files with Lean

The Lean project in `proofs/` builds an executable, `check_certificate`,
that reads one of these files, recomputes every claim over the
rationals, and exits with code 0 when it accepts, 1 when it rejects, and
2 when the file cannot be decoded. It is run as

``` sh
cd proofs
lake exe cache get                 # once: prebuilt Mathlib
lake build check_certificate       # once
lake exe check_certificate ../vignettes/10_exact_arithmetic_and_certificates/_out/bd_w1.json
```

The cell below runs the built executable when it is present, and says so
when it is not. The output recorded in the rendered copies of this
vignette was produced with the executable built.

``` julia
checker = joinpath(@__DIR__, "..", "..", "proofs", ".lake", "build", "bin",
                   "check_certificate")

function lean_check(file)
    println("check_certificate ", basename(file))
    if !isfile(checker)
        println("  not run: build the checker first (see proofs/README.md)")
        return nothing
    end
    buffer = IOBuffer()
    process = run(pipeline(ignorestatus(`$checker $file`); stdout=buffer, stderr=buffer))
    foreach(line -> println("  ", line), split(chomp(String(take!(buffer))), '\n'))
    println("  exit code: ", process.exitcode)
    return nothing
end

lean_check(certificate)
```

    check_certificate bd_w1.json
      schema: decision-certificate/1
      actions: 3, states: 3
      prior nonnegative: ok
      prior sums to one: ok
      expected_utility: ok
      act_now_value: ok
      bayes_actions: ok
      regret: ok
      expected_regret: ok
      max_regret: ok
      minimax_regret_actions: ok
      evpi: ok
      likelihood column-stochastic: ok
      value_with_information: ok
      evsi: ok
      accepted: yes
      theorem Certificate.certificate_spec: applies
      exit code: 0

``` julia
lean_check(joinpath(out, "bd_w0.json"))
```

    check_certificate bd_w0.json
      schema: decision-certificate/1
      actions: 3, states: 3
      prior nonnegative: ok
      prior sums to one: ok
      expected_utility: ok
      act_now_value: ok
      bayes_actions: ok
      regret: ok
      expected_regret: ok
      max_regret: ok
      minimax_regret_actions: ok
      evpi: ok
      likelihood column-stochastic: ok
      value_with_information: ok
      evsi: ok
      accepted: yes
      theorem Certificate.certificate_spec: applies
      exit code: 0

``` julia
lean_check(garbling)
```

    check_certificate pooled_is_garbling_of_swab.json
      schema: garbling-witness/1
      states: 3, signals: 2, garbled signals: 2
      shapes: ok
      experiment column-stochastic: ok
      garbling column-stochastic: ok
      garbled = garbling * experiment: ok
      accepted: yes
      theorem Certificate.RawGarbling.accepts_sound: applies
      exit code: 0

``` julia
lean_check(admissible)
```

    check_certificate disinfect_is_e_admissible.json
      schema: e-admissible-witness/1
      actions: 3, states: 3, extreme points: 3, action: 2
      accepted: yes
      theorem Certificate.RawEAdmissible.accepts_sound: applies
      exit code: 0

### A wrong claim is rejected

To see the checker refuse, change one claim by hand. Here the EVSI in
the certificate is replaced by a rounder number.

``` julia
text = read(certificate, String)
claimed = "\"evsi\": \"" * exact(evsi(problem, swab)) * "\""
occursin(claimed, text)
```

    true

``` julia
tampered = joinpath(out, "bd_w1_tampered.json")
write(tampered, replace(text, claimed => "\"evsi\": \"31\""))
lean_check(tampered)
```

    check_certificate bd_w1_tampered.json
      schema: decision-certificate/1
      actions: 3, states: 3
      prior nonnegative: ok
      prior sums to one: ok
      expected_utility: ok
      act_now_value: ok
      bayes_actions: ok
      regret: ok
      expected_regret: ok
      max_regret: ok
      minimax_regret_actions: ok
      evpi: ok
      likelihood column-stochastic: ok
      value_with_information: ok
      evsi: FAIL
      accepted: no
      exit code: 1

## What is and is not certified

An accepted certificate says that the claimed numbers are the correct
expected utilities, regrets and values of information *for the table,
prior and likelihood in the file*. It says nothing about whether the
table is a good description of the pond. The numbers in this vignette
are illustrative, and a certificate of an illustrative table certifies
arithmetic about an illustration.

The certificate covers a single prior. The witnesses cover one garbling
and one E-admissible action each. Deficiency, robust value of
information and the `2 B ε` bound are tested in Julia and are not
covered by the Lean checker.

## What to take away

- Integer and `Rational` inputs give exact results: no rounding, true
  ties, and identities that hold with `==`.
- `export_decision_certificate` writes a problem and its claimed
  numbers; `export_garbling_witness` and `export_eadmissible_witness`
  write the witnesses found by linear programme. All three refuse
  floating-point input, and the witness exporters refuse to write a
  false claim.
- `lake exe check_certificate FILE.json` in `proofs/` recomputes a file
  independently and exits 0, 1 or 2 for accepted, rejected and
  malformed.
- A certificate certifies the arithmetic, not the inputs.

## The series

This is the last vignette of the series. The first is [A first decision
table](01_first_decision_table.md).
