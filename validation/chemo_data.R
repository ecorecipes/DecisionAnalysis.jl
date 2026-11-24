# Writes the chemotherapy example bundled with the R package `voi` to two CSV files in
# the directory given as the first argument, for the gated test (DECISIONANALYSIS_R=1)
# in test/test_voi_reference.jl. The data stay in the installed package: they are not
# committed to this repository.
library(voi)
dir <- commandArgs(trailingOnly = TRUE)[1]
write.csv(chemo_nb, file.path(dir, "chemo_nb.csv"), row.names = FALSE)
write.csv(chemo_pars, file.path(dir, "chemo_pars.csv"), row.names = FALSE)
cat(as.character(packageVersion("voi")), "\n")
