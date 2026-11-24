# Prints the estimates of this package beside the reference values of the R package
# `voi` in test/fixtures/voi/voi_reference.json. Run from the package root in an
# environment that has DecisionAnalysis, GAM and JSON (the test environment does):
#
#   julia --project=<environment> validation/compare_voi.jl [directory with chemo_*.csv]
#
# With a directory written by `Rscript validation/chemo_data.R <directory>` the
# chemotherapy example is compared as well.
using DecisionAnalysis, GAM, JSON

const ROOT = joinpath(@__DIR__, "..")
const REF = JSON.parsefile(joinpath(ROOT, "test", "fixtures", "voi", "voi_reference.json"))

function read_csv(path)
    lines = readlines(path)
    names = Symbol.(strip.(split(lines[1], ","), '"'))
    data = [parse.(Float64, split(line, ",")) for line in lines[2:end]]
    return NamedTuple{Tuple(names)}(Tuple([row[j] for row in data]
                                          for j in eachindex(names)))
end

function row(label, ours, theirs)
    value = ours isa VOIEstimate ? ours.value : ours
    se = ours isa VOIEstimate ? string(round(ours.standard_error; sigdigits=3)) : "-"
    return println(rpad(label, 34), rpad(round(value; sigdigits=8), 14),
                   rpad(round(theirs; sigdigits=8), 14), rpad(se, 10),
                   round(value - theirs; sigdigits=3))
end
function header(title)
    return println("\n", title, "\n", rpad("quantity", 34), rpad("ours", 14),
                   rpad("voi", 14), rpad("our s.e.", 10), "difference")
end

s = read_csv(joinpath(ROOT, "test", "fixtures", "voi", "normal_sample.csv"))
n = length(s.p1)
function problem(new)
    return DecisionProblem(PayoffTable(permutedims(hcat(zeros(n), new));
                                       actions=[:reference, :new]), SampleBelief(s))
end
normal = problem(s.p1 .- s.p2)
r = REF["normal"]
header("Normal model (2000 shared draws)")
row("EVPI", evpi(normal, SampleAverage()), r["evpi"])
row("EVPPI p1, GAM", evppi(normal, :p1, GAMRegression()), r["p1_gam"])
row("EVPPI p2, GAM", evppi(normal, :p2, GAMRegression()), r["p2_gam"])
row("EVPPI p1, spline", evppi(normal, :p1, SplineRegression()), r["p1_gam"])
row("EVPPI p1, GP", evppi(normal, :p1, GaussianProcessRegression(; max_points=2000)),
    r["p1_gp"])
row("EVPPI p2, GP", evppi(normal, :p2, GaussianProcessRegression(; max_points=2000)),
    r["p2_gp"])
row("EVPPI p1, so 20 blocks", evppi(normal, :p1, StrongOakley(; blocks=20)), r["p1_so_20"])
row("EVPPI p2, so 20 blocks", evppi(normal, :p2, StrongOakley(; blocks=20)), r["p2_so_20"])
row("EVPPI p1, so 50 blocks", evppi(normal, :p1, StrongOakley(; blocks=50)), r["p1_so_50"])
row("EVPPI p1, sal 1", evppi(normal, :p1, Sadatsafavi()), r["p1_sal_1"])
row("EVPPI p2, sal 1", evppi(normal, :p2, Sadatsafavi()), r["p2_sal_1"])
row("EVPPI p1, sal 2", evppi(normal, :p1, Sadatsafavi(; separators=2)), r["p1_sal_2"])
row("EVPPI p2, sal 2", evppi(normal, :p2, Sadatsafavi(; separators=2)), r["p2_sal_2"])
row("EVPPI p1+p2, GAM tensor", evppi(normal, (:p1, :p2), GAMRegression(; tensor=true)),
    r["p1_p2_gam"])
row("EVPPI p1+p2, GP",
    evppi(normal, (:p1, :p2), GaussianProcessRegression(; max_points=2000)), r["p1_p2_gp"])
# EVSI of m observations of p1 with known sd 1; closed forms 0.0723251 and 0.0821176.
using Random
for m in (10, 100)
    xbar = s.p1 .+ randn(Xoshiro(m), n) ./ sqrt(m)
    row("EVSI p1 n=$m, regression", evsi(normal, SummaryStatistics((; xbar))),
        r["evsi_p1_n$(m)_gam"])
    study = importance_loglikelihoods((rng, θ) -> θ.p1 + randn(rng) / sqrt(m),
                                      (x, θ) -> -m * (x - θ.p1)^2 / 2, normal.belief;
                                      datasets=1000, seed=m)
    row("EVSI p1 n=$m, importance", evsi(normal, study), r["evsi_p1_n$(m)_is"])
    row("EVSI p1 n=$m, Gaussian approx.",
        evsi(normal, GaussianApproximation(:p1; n0=1, n=m)), r["evsi_p1_n$(m)_gam"])
end

six = problem(s.p1 .- s.p2 .+ 0.5 .* (s.x3 .+ s.x4 .+ s.x5 .+ s.x6))
r = REF["six"]
header("Six-parameter normal model (same draws)")
gp = GaussianProcessRegression(; max_points=2000)
row("EVPI", evpi(six, SampleAverage()), r["evpi"])
row("EVPPI p1,x3..x6, default (GP)", evppi(six, (:p1, :x3, :x4, :x5, :x6)),
    r["p1_x3_x4_x5_x6_default"])
row("EVPPI p1,x3..x6, GP all points", evppi(six, (:p1, :x3, :x4, :x5, :x6), gp),
    r["p1_x3_x4_x5_x6_gp"])
row("EVPPI p2,x3..x6, GP all points", evppi(six, (:p2, :x3, :x4, :x5, :x6), gp),
    r["p2_x3_x4_x5_x6_default"])
row("EVPPI all six, GP all points", evppi(six, (:p1, :p2, :x3, :x4, :x5, :x6), gp),
    r["all_six_gp"])

if !isempty(ARGS)
    nb = read_csv(joinpath(ARGS[1], "chemo_nb.csv"))
    pars = read_csv(joinpath(ARGS[1], "chemo_pars.csv"))
    Q = permutedims(hcat(nb.SoC, nb.Novel))
    chemo = DecisionProblem(PayoffTable(Q; actions=[:SoC, :Novel]), SampleBelief(pars))
    first = 1:1500
    head = DecisionProblem(PayoffTable(Q[:, first]; actions=[:SoC, :Novel]),
                           SampleBelief(map(c -> c[first], pars)))
    r = REF["chemotherapy"]
    header("voi chemotherapy example (net benefit in pounds; a validation target only)")
    row("EVPI", evpi(chemo, SampleAverage()), r["evpi"])
    for name in (:p_side_effects_t1, :logor_side_effects)
        row("EVPPI $name, GAM", evppi(chemo, name, GAMRegression()), r["$(name)_gam"])
        row("EVPPI $name, so 50", evppi(chemo, name, StrongOakley(; blocks=50)),
            r["$(name)_so_50"])
        row("EVPPI $name, sal 1", evppi(chemo, name, Sadatsafavi()), r["$(name)_sal_1"])
        row("EVPPI $name, sal 2", evppi(chemo, name, Sadatsafavi(; separators=2)),
            r["$(name)_sal_2"])
    end
    row("EVPPI logor, GP first 1500",
        evppi(head, :logor_side_effects, GaussianProcessRegression()),
        r["logor_side_effects_gp_first1500"])
    row("EVPPI u_hospital, GAM", evppi(chemo, :u_hospital, GAMRegression()),
        r["u_hospital_gam"])
    row("EVPPI u_hospital, sal 1", evppi(chemo, :u_hospital, Sadatsafavi()),
        r["u_hospital_sal_1"])
    row("EVPPI pair, GAM tensor",
        evppi(chemo, (:p_side_effects_t1, :logor_side_effects),
              GAMRegression(; tensor=true)), r["pair_gam"])
    group = (:p_side_effects_t1, :p_side_effects_t2, :logor_side_effects,
             :p_hospitalised_total, :p_died)
    row("EVPPI group of 5, GP first 1500", evppi(head, group),
        r["group_of_five_gp_first1500"])
end
