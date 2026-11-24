# Reference values from the R package `voi` (Heath, Jackson and others; CRAN) for
# test/test_voi_reference.jl. Run from the package root:
#
#   Rscript validation/voi_reference.R
#
# It reads test/fixtures/voi/normal_sample.csv (written by validation/make_sample.jl), so
# that R and Julia work on the same draws, and uses the chemotherapy example bundled
# with voi (`chemo_nb`, `chemo_pars`: net benefits in pounds at a willingness to pay of
# 20,000 pounds per QALY), and writes test/fixtures/voi/voi_reference.json. The
# chemotherapy data are not copied into this repository (voi is GPL-3): with
# DECISIONANALYSIS_R=1 the test fetches them from the installed package through
# validation/chemo_data.R. Money appears here only as a published validation target.
library(voi)
set.seed(1)
out <- "test/fixtures/voi/voi_reference.json"
num <- function(x) sprintf("%.12g", x)
entry <- function(name, x) sprintf("    \"%s\": %s", name, num(x))
block <- function(name, entries) {
  sprintf("  \"%s\": {\n%s\n  }", name, paste(entries, collapse = ",\n"))
}
ev <- function(nb, inputs, pars, ...) evppi(nb, inputs, pars = pars, ...)$evppi[1]

s <- read.csv("test/fixtures/voi/normal_sample.csv")
nb <- data.frame(reference = 0, new = s$p1 - s$p2)
normal <- c(
  entry("evpi", evpi(nb)),
  entry("p1_gam", ev(nb, s, "p1", method = "gam")),
  entry("p2_gam", ev(nb, s, "p2", method = "gam")),
  entry("p1_gp", ev(nb, s, "p1", method = "gp")),
  entry("p2_gp", ev(nb, s, "p2", method = "gp")),
  entry("p1_so_20", ev(nb, s, "p1", method = "so", n.blocks = 20)),
  entry("p2_so_20", ev(nb, s, "p2", method = "so", n.blocks = 20)),
  entry("p1_so_50", ev(nb, s, "p1", method = "so", n.blocks = 50)),
  entry("p1_sal_1", ev(nb, s, "p1", method = "sal")),
  entry("p2_sal_1", ev(nb, s, "p2", method = "sal")),
  entry("p1_sal_2", ev(nb, s, "p1", method = "sal", n.seps = 2)),
  entry("p2_sal_2", ev(nb, s, "p2", method = "sal", n.seps = 2)),
  entry("p1_p2_gam", ev(nb, s, c("p1", "p2"), method = "gam")),
  entry("p1_p2_gp", ev(nb, s, c("p1", "p2"), method = "gp")),
  # EVSI of n observations of p1 with known standard deviation 1 (closed forms
  # 0.0723251 at n = 10 and 0.0821176 at n = 100).
  entry("evsi_p1_n10_gam", evsi(nb, s, study = "normal_known", n = 10, pars = "p1",
                                 aux_pars = list(sd = 1), method = "gam")$evsi[1]),
  entry("evsi_p1_n100_gam", evsi(nb, s, study = "normal_known", n = 100, pars = "p1",
                                  aux_pars = list(sd = 1), method = "gam")$evsi[1]),
  entry("evsi_p1_n10_is", evsi(nb, s, study = "normal_known", n = 10, pars = "p1",
                                aux_pars = list(sd = 1), method = "is", nsim = 1000)$evsi[1]),
  entry("evsi_p1_n100_is", evsi(nb, s, study = "normal_known", n = 100, pars = "p1",
                                 aux_pars = list(sd = 1), method = "is", nsim = 1000)$evsi[1])
)

nb6 <- data.frame(reference = 0, new = s$p1 - s$p2 + 0.5 * (s$x3 + s$x4 + s$x5 + s$x6))
five <- c("p1", "x3", "x4", "x5", "x6")
other <- c("p2", "x3", "x4", "x5", "x6")
six <- c(
  entry("evpi", evpi(nb6)),
  entry("p1_x3_x4_x5_x6_default", ev(nb6, s, five)),
  entry("p2_x3_x4_x5_x6_default", ev(nb6, s, other)),
  entry("p1_x3_x4_x5_x6_gp", ev(nb6, s, five, method = "gp")),
  entry("all_six_gp", ev(nb6, s, c("p1", "p2", "x3", "x4", "x5", "x6"), method = "gp"))
)

group <- c("p_side_effects_t1", "p_side_effects_t2", "logor_side_effects",
           "p_hospitalised_total", "p_died")
# voi's Gaussian process on all 10,000 draws (it conditions on `maxSample = 5000` of
# them) exceeded 18 Gb of memory here, and voi 1.0.3 fails with `maxSample` below the
# number of draws ("number of items to replace is not a multiple of replacement
# length"), so the Gaussian-process comparisons use the first 1500 draws on both sides.
first <- 1:1500
chemo <- c(
  entry("evpi", evpi(chemo_nb)),
  entry("draws", nrow(chemo_nb)),
  entry("p_side_effects_t1_gam", ev(chemo_nb, chemo_pars, "p_side_effects_t1", method = "gam")),
  entry("p_side_effects_t1_so_50", ev(chemo_nb, chemo_pars, "p_side_effects_t1", method = "so", n.blocks = 50)),
  entry("p_side_effects_t1_sal_1", ev(chemo_nb, chemo_pars, "p_side_effects_t1", method = "sal")),
  entry("p_side_effects_t1_sal_2", ev(chemo_nb, chemo_pars, "p_side_effects_t1", method = "sal", n.seps = 2)),
  entry("logor_side_effects_gam", ev(chemo_nb, chemo_pars, "logor_side_effects", method = "gam")),
  entry("logor_side_effects_so_50", ev(chemo_nb, chemo_pars, "logor_side_effects", method = "so", n.blocks = 50)),
  entry("logor_side_effects_sal_1", ev(chemo_nb, chemo_pars, "logor_side_effects", method = "sal")),
  entry("logor_side_effects_sal_2", ev(chemo_nb, chemo_pars, "logor_side_effects", method = "sal", n.seps = 2)),
  entry("logor_side_effects_gp_first1500", ev(chemo_nb[first, ], chemo_pars[first, ], "logor_side_effects", method = "gp")),
  entry("u_hospital_gam", ev(chemo_nb, chemo_pars, "u_hospital", method = "gam")),
  entry("u_hospital_sal_1", ev(chemo_nb, chemo_pars, "u_hospital", method = "sal")),
  entry("pair_gam", ev(chemo_nb, chemo_pars, c("p_side_effects_t1", "logor_side_effects"), method = "gam")),
  entry("group_of_five_gp_first1500", ev(chemo_nb[first, ], chemo_pars[first, ], group))
)

sink(out)
cat("{\n")
cat(sprintf("  \"provenance\": {\n    \"R\": \"%s\",\n    \"voi\": \"%s\",\n    \"mgcv\": \"%s\",\n    \"script\": \"validation/voi_reference.R\",\n    \"sample\": \"test/fixtures/voi/normal_sample.csv\"\n  },\n",
            R.version.string, as.character(packageVersion("voi")),
            as.character(packageVersion("mgcv"))))
cat(block("normal", normal), ",\n", sep = "")
cat(block("six", six), ",\n", sep = "")
cat(block("chemotherapy", chemo), "\n", sep = "")
cat("}\n")
sink()
