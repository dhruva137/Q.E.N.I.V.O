# Builds the known LP and the two-binary MILP through the .Call binding.
# From this directory, after R CMD SHLIB src/qenivo_r.c:
#   Rscript example.R

dyn.load(Sys.getenv("QENIVO_R_DLL", "src/qenivo.dll"))
source("R/qenivo.R")

bound <- bounded_lp()
stopifnot(abs(bound$objective + 1) < 1e-8, abs(bound$x[[1]] - 1) < 1e-8)
cat("optimum", bound$objective, "\n")

milp <- two_binary()
stopifnot(abs(milp$objective + 2) < 1e-6, abs(milp$x[[1]]) < 1e-6, abs(milp$x[[2]] - 1) < 1e-6)
cat("milp", milp$objective, "\n")
