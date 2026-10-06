// Exact rational Gaussian elimination (an LU factorisation with the multipliers kept).
#ifndef QENIVO_EXACT_LU_HPP
#define QENIVO_EXACT_LU_HPP

#include "rational.hpp"

#include <stdexcept>
#include <vector>

namespace qenivo {
namespace exact {

class Singular : public std::runtime_error {
public:
    explicit Singular(const std::string& what) : std::runtime_error(what) {}
};

// Solve A x = b. A is row-major and square. Throws Singular if A has no inverse.
std::vector<Rat> solve_exact(const std::vector<std::vector<Rat>>& a, const std::vector<Rat>& b);

}  // namespace exact
}  // namespace qenivo

#endif
