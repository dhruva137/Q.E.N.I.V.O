# .Call interface. dyn.load the shared library built from src/qenivo_r.c, after
# QENIVO_C_DLL names the C ABI library and QENIVO_SRC names the Python package.

#' Solve min -x on 0 <= x <= 1. Returns a list with objective and x.
bounded_lp <- function() {
  .Call(qenivo_bounded_lp, NULL)
}

#' Solve min -x - 2y, x + y <= 1, x and y binary.
two_binary <- function() {
  .Call(qenivo_two_binary, NULL)
}
