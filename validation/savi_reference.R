# Reference values for the test data of SAVI (Sheffield Accelerated Value of
# Information, https://github.com/Sheffield-Accelerated-VoI/SAVI, BSD 3-clause):
# 1000 draws of the 19 parameters, the costs and the effects of a two-option model.
# The data are copied unchanged into test/fixtures/savi/ with SAVI's licence. This
# script is ours: it fits the regressions that SAVI fits (one smooth of a single
# parameter, a tensor product for a group) with mgcv and writes
# test/fixtures/savi/savi_reference.json. Money appears only as the unit of the
# source: net benefit = lambda * effects - costs at SAVI's default lambda.
#
#   Rscript validation/savi_reference.R
suppressPackageStartupMessages({library(mgcv); library(jsonlite)})
dir <- file.path("test", "fixtures", "savi")
costs <- as.matrix(read.csv(file.path(dir, "costs.csv")))
effects <- as.matrix(read.csv(file.path(dir, "effects.csv")))
params <- read.csv(file.path(dir, "parameters.csv"))
lambda <- 20000
nb <- lambda * effects - costs
inb <- nb[, 2] - nb[, 1]
value <- function(fitted) mean(pmax(0, fitted)) - max(0, mean(fitted))
single <- sapply(names(params), function(p) {
  x <- params[[p]]
  if (var(x) == 0) return(0)
  value(fitted(gam(inb ~ s(x))))
})
groups <- list(c("theta5", "theta14"), c("theta6", "theta15"), c("theta7", "theta16"))
grouped <- lapply(groups, function(g) {
  f <- as.formula(paste0("inb ~ te(", paste(g, collapse = ", "), ")"))
  list(parameters = g, evppi = value(fitted(gam(f, data = params))))
})
out <- list(
  source = "SAVI test_data/brennan1000 (costs.csv, effects.csv, parameters.csv), BSD 3-clause, copyright 2014 The University of Sheffield",
  script = "validation/savi_reference.R",
  versions = list(R = R.version.string, mgcv = as.character(packageVersion("mgcv"))),
  lambda = lambda,
  draws = nrow(nb),
  expected_net_benefit = unname(colMeans(nb)),
  evpi = mean(pmax(nb[, 1], nb[, 2])) - max(colMeans(nb)),
  probability_best = c(mean(nb[, 1] >= nb[, 2]), mean(nb[, 2] >= nb[, 1])),
  evppi_single = as.list(single),
  evppi_groups = grouped
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 12, pretty = TRUE),
           file.path(dir, "savi_reference.json"))
