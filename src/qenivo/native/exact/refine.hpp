// Iterative refinement of a rational linear system to an exact solution.
//
// A. Gleixner, D. E. Steffy and K. Wolter, "Iterative refinement for linear
// programming", Mathematical Programming Computation, 2016.
// The residual correction is the one in N. J. Higham, Accuracy and Stability
// of Numerical Algorithms, SIAM. A reconstructed rational is accepted only
// when the exact residual is zero. Otherwise solve_exact is the fallback.
#ifndef QENIVO_EXACT_REFINE_HPP
#define QENIVO_EXACT_REFINE_HPP

#include "lu.hpp"

#include <optional>
#include <string>
#include <vector>

namespace qenivo {
namespace exact {

struct Solved {
    std::vector<Rat> x;
    std::string method;  // "iterative_refinement" or "exact_lu"
};

// Empty optional only when the matrix is singular.
std::optional<Solved> solve_system(const std::vector<std::vector<Rat>>& a, const std::vector<Rat>& b);

// Refinement alone. Empty when reconstruction does not land on an exact solution.
std::optional<std::vector<Rat>> solve_refined(const std::vector<std::vector<Rat>>& a,
                                              const std::vector<Rat>& b);

}  // namespace exact
}  // namespace qenivo

#endif
