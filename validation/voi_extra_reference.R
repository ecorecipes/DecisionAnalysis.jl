# Reference values from the R package `voi` for the methods added after the first
# fixture: multivariate adaptive regression splines (`earth`), moment matching, the
# variance-based (estimation) measures, the population value and the net benefit of
# sampling. Run from the package root:
#
#   Rscript validation/voi_extra_reference.R
#
# It reads test/fixtures/voi/normal_sample.csv (the draws R and Julia share) and writes
# test/fixtures/voi/voi_extra_reference.json. See validation/README.md.
library(voi)
set.seed(1)
out <- "test/fixtures/voi/voi_extra_reference.json"
num <- function(x) sprintf("%.12g", x)
entry <- function(name, x) sprintf("    \"%s\": %s", name, num(x))
vec <- function(name, x) sprintf("    \"%s\": [%s]", name, paste(num(x), collapse = ", "))
block <- function(name, entries) {
  sprintf("  \"%s\": {\n%s\n  }", name, paste(entries, collapse = ",\n"))
}
ev <- function(nb, inputs, pars, ...) evppi(nb, inputs, pars = pars, ...)$evppi[1]

s <- read.csv("test/fixtures/voi/normal_sample.csv")
nb <- data.frame(reference = 0, new = s$p1 - s$p2)
y <- s$p1 - s$p2

earth <- c(
  entry("p1_earth", ev(nb, s, "p1", method = "earth")),
  entry("p2_earth", ev(nb, s, "p2", method = "earth")),
  entry("p1_p2_earth", ev(nb, s, c("p1", "p2"), method = "earth"))
)

# Moment matching for n observations of p1 with known standard deviation 1; the prior
# of p1 is N(1, 1) (closed forms 0.0723251 at n = 10 and 0.0821176 at n = 100).
model_fn <- function(p1, p2) c(0, p1 - p2)
par_fn <- function(n) data.frame(p1 = rnorm(n, 1, 1), p2 = rnorm(n, 0, 2))
mm <- function(n) {
  evsi(nb, s, study = "normal_known", n = n, pars = "p1", method = "mm", Q = 30,
       aux_pars = list(sd = 1), model_fn = model_fn, par_fn = par_fn,
       analysis_args = list(prior_mean = 1, prior_sd = 1, sampling_sd = 1, niter = 1000))$evsi[1]
}
moment <- c(entry("evsi_p1_n10_mm", mm(10)), entry("evsi_p1_n100_mm", mm(100)))

# Estimation: expected reductions in the variance of y = p1 - p2 (closed forms 1 and 4
# for learning p1 and p2, and n / (n + 1) for n observations of p1).
variance <- c(
  entry("var_y", var(y)),
  entry("evppivar_p1_gam", evppivar(y, s, pars = "p1", method = "gam")$evppi[1]),
  entry("evppivar_p2_gam", evppivar(y, s, pars = "p2", method = "gam")$evppi[1]),
  entry("evppivar_p1_earth", evppivar(y, s, pars = "p1", method = "earth")$evppi[1]),
  entry("evsivar_p1_n10_gam", evsivar(y, s, study = "normal_known", n = 10, pars = "p1",
                                      aux_pars = list(sd = 1))$evsi[1]),
  entry("evsivar_p1_n100_gam", evsivar(y, s, study = "normal_known", n = 100, pars = "p1",
                                       aux_pars = list(sd = 1))$evsi[1])
)

# Population value and the net benefit of sampling, on a fixed EVSI curve so that
# nothing here depends on a random number.
n <- c(10, 20, 30, 40, 60, 80, 100, 150, 200, 300)
curve <- data.frame(n = n, k = 1, evsi = 0.08 * n / (n + 12))
e <- enbs(curve, costs_setup = c(5, 15), costs_pp = c(0.04, 0.06), pop = 1000, time = 10,
          dis = 0.035)
opt <- attr(e, "enbsmax")
smooth <- enbs_opt(e, smooth = TRUE)
design <- c(
  vec("n", n),
  vec("evsi", curve$evsi),
  entry("pop_voi_dis", pop_voi(2, 10, 5, 0.035)),
  entry("pop_voi_zero", pop_voi(2, 10, 5, 0)),
  vec("enbs", e$enbs),
  vec("sd", e$sd),
  vec("pce", e$pce),
  entry("enbsmax", opt$enbsmax),
  entry("nmax", opt$nmax),
  entry("nlower", opt$nlower),
  entry("nupper", opt$nupper),
  entry("smooth_enbsmax", smooth$enbsmax),
  entry("smooth_nmax", smooth$nmax),
  entry("smooth_nlower", smooth$nlower),
  entry("smooth_nupper", smooth$nupper)
)

sink(out)
cat("{\n")
cat(sprintf("  \"provenance\": {\n    \"R\": \"%s\",\n    \"voi\": \"%s\",\n    \"earth\": \"%s\",\n    \"script\": \"validation/voi_extra_reference.R\",\n    \"sample\": \"test/fixtures/voi/normal_sample.csv\"\n  },\n",
            R.version.string, as.character(packageVersion("voi")),
            as.character(packageVersion("earth"))))
cat(block("earth", earth), ",\n", sep = "")
cat(block("moment_matching", moment), ",\n", sep = "")
cat(block("variance", variance), ",\n", sep = "")
cat(block("design", design), "\n", sep = "")
cat("}\n")
sink()
