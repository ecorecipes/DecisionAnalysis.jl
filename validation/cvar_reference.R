# Conditional value at risk and value at risk of the worked examples by the R package
# `cvar` (Boshnakov), for test/test_validation.jl. Run from the package root:
#
#   Rscript validation/cvar_reference.R
#
# `cvar` works with losses and returns positive numbers for bad tails: for a payoff X
# with quantile function q, `ES(q, p_loss = a, dist.type = "qf")` is −(1/a) ∫₀^a q(u) du
# and `VaR` is −q(a). This package works with utilities, so `CVaR(a)` is minus the
# first and `Quantile(a)` minus the second. `ES` integrates the quantile function
# numerically, so the comparison is to a tolerance.
library(cvar)
library(jsonlite)
num <- function(s) sapply(strsplit(as.character(s), "/"), function(v) {
  as.numeric(v[1]) / ifelse(length(v) == 2, as.numeric(v[2]), 1)
})
examples <- fromJSON("test/fixtures/worked_examples.json", simplifyVector = FALSE)$examples
levels <- c(0.05, 0.25, 0.5, 0.9)
rows <- list()
for (ex in examples) {
  prior <- num(unlist(ex$prior))
  for (a in seq_along(ex$payoff)) {
    x <- num(unlist(ex$payoff[[a]]))
    o <- order(x)
    xs <- x[o]
    cum <- cumsum(prior[o])
    qf <- function(u) xs[pmin(findInterval(u, cum, left.open = TRUE) + 1, length(xs))]
    for (level in levels) {
      rows[[length(rows) + 1]] <- list(
        example = ex$name, action = a, level = level,
        expected_shortfall = ES(qf, p_loss = level, dist.type = "qf"),
        value_at_risk = VaR(qf, p_loss = level, dist.type = "qf"))
    }
  }
}
out <- list(
  provenance = list(R = R.version.string, cvar = as.character(packageVersion("cvar")),
                    script = "validation/cvar_reference.R"),
  convention = "losses: expected_shortfall = -CVaR(level), value_at_risk = -Quantile(level)",
  rows = rows)
dir.create("test/fixtures/validation", showWarnings = FALSE, recursive = TRUE)
write(toJSON(out, auto_unbox = TRUE, digits = 12), "test/fixtures/validation/cvar_reference.json")
cat("wrote test/fixtures/validation/cvar_reference.json:", length(rows), "rows\n")
