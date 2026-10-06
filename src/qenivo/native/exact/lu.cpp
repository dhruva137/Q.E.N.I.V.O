#include "lu.hpp"

#include <stdexcept>
#include <utility>

namespace qenivo {
namespace exact {

std::vector<Rat> solve_exact(const std::vector<std::vector<Rat>>& a, const std::vector<Rat>& b) {
    const int m = static_cast<int>(a.size());
    if (m == 0) {
        return {};
    }
    if (static_cast<int>(b.size()) != m) {
        throw std::invalid_argument("row count does not match the right-hand side");
    }
    std::vector<std::vector<Rat>> mat(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
        if (static_cast<int>(a[static_cast<std::size_t>(i)].size()) != m) {
            throw std::invalid_argument("exact solve needs a square matrix");
        }
        mat[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)];
        mat[static_cast<std::size_t>(i)].push_back(b[static_cast<std::size_t>(i)]);
    }
    for (int col = 0; col < m; ++col) {
        int pivot = col;
        for (int row = col + 1; row < m; ++row) {
            if (abs_greater(mat[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)],
                            mat[static_cast<std::size_t>(pivot)][static_cast<std::size_t>(col)])) {
                pivot = row;
            }
        }
        if (mat[static_cast<std::size_t>(pivot)][static_cast<std::size_t>(col)].is_zero()) {
            throw Singular("zero pivot");
        }
        if (pivot != col) {
            std::swap(mat[static_cast<std::size_t>(col)], mat[static_cast<std::size_t>(pivot)]);
        }
        const Rat piv = mat[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)];
        for (int row = col + 1; row < m; ++row) {
            Rat& lead = mat[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
            if (lead.is_zero()) {
                continue;
            }
            const Rat factor = lead / piv;
            lead = Rat(0);
            for (int j = col + 1; j <= m; ++j) {
                Rat& entry = mat[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)];
                entry = entry - factor * mat[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)];
            }
        }
    }
    std::vector<Rat> x(static_cast<std::size_t>(m), Rat(0));
    for (int i = m - 1; i >= 0; --i) {
        Rat acc = mat[static_cast<std::size_t>(i)][static_cast<std::size_t>(m)];
        for (int j = i + 1; j < m; ++j) {
            acc = acc - mat[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
        }
        const Rat& diag = mat[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
        if (diag.is_zero()) {
            throw Singular("zero diagonal");
        }
        x[static_cast<std::size_t>(i)] = acc / diag;
    }
    return x;
}

}  // namespace exact
}  // namespace qenivo
