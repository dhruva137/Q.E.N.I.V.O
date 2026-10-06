#include "refine.hpp"

#include <cmath>
#include <utility>

namespace qenivo {
namespace exact {
namespace {

bool solve_double(const std::vector<std::vector<double>>& a, const std::vector<double>& b,
                  std::vector<double>& x) {
    const int n = static_cast<int>(a.size());
    std::vector<std::vector<double>> mat(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n + 1)));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            mat[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        }
        mat[static_cast<std::size_t>(i)][static_cast<std::size_t>(n)] = b[static_cast<std::size_t>(i)];
    }
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        for (int row = col + 1; row < n; ++row) {
            if (std::fabs(mat[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]) >
                std::fabs(mat[static_cast<std::size_t>(pivot)][static_cast<std::size_t>(col)])) {
                pivot = row;
            }
        }
        const double piv = mat[static_cast<std::size_t>(pivot)][static_cast<std::size_t>(col)];
        if (!std::isfinite(piv) || piv == 0.0) {
            return false;
        }
        if (pivot != col) {
            std::swap(mat[static_cast<std::size_t>(col)], mat[static_cast<std::size_t>(pivot)]);
        }
        const double pk = mat[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)];
        for (int row = col + 1; row < n; ++row) {
            const double factor = mat[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] / pk;
            mat[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = 0.0;
            for (int j = col + 1; j <= n; ++j) {
                mat[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)] -=
                    factor * mat[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)];
            }
        }
    }
    x.assign(static_cast<std::size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double acc = mat[static_cast<std::size_t>(i)][static_cast<std::size_t>(n)];
        for (int j = i + 1; j < n; ++j) {
            acc -= mat[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
        }
        const double diag = mat[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
        if (!std::isfinite(diag) || diag == 0.0) {
            return false;
        }
        x[static_cast<std::size_t>(i)] = acc / diag;
        if (!std::isfinite(x[static_cast<std::size_t>(i)])) {
            return false;
        }
    }
    return true;
}

std::vector<Rat> residual(const std::vector<std::vector<Rat>>& a, const std::vector<Rat>& x,
                          const std::vector<Rat>& b) {
    std::vector<Rat> out(b.size(), Rat(0));
    for (std::size_t i = 0; i < b.size(); ++i) {
        Rat acc = b[i];
        for (std::size_t j = 0; j < x.size(); ++j) {
            acc = acc - a[i][j] * x[j];
        }
        out[i] = acc;
    }
    return out;
}

bool exact_zero(const std::vector<Rat>& values) {
    for (const Rat& value : values) {
        if (!value.is_zero()) {
            return false;
        }
    }
    return true;
}

}  // namespace

std::optional<std::vector<Rat>> solve_refined(const std::vector<std::vector<Rat>>& a, const std::vector<Rat>& b) {
    const int n = static_cast<int>(a.size());
    if (n == 0) {
        return std::vector<Rat>{};
    }
    std::vector<std::vector<double>> af(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n)));
    std::vector<double> bf(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            af[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].to_double();
        }
        bf[static_cast<std::size_t>(i)] = b[static_cast<std::size_t>(i)].to_double();
        if (!std::isfinite(bf[static_cast<std::size_t>(i)])) {
            return std::nullopt;
        }
    }
    std::vector<double> x;
    if (!solve_double(af, bf, x)) {
        return std::nullopt;
    }
    for (int round = 0; round < 6; ++round) {
        std::vector<Rat> dyadic(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(x[static_cast<std::size_t>(i)])) {
                return std::nullopt;
            }
            dyadic[static_cast<std::size_t>(i)] = rat_from_double(x[static_cast<std::size_t>(i)]);
        }
        const std::vector<Rat> res = residual(a, dyadic, b);
        if (exact_zero(res)) {
            return dyadic;
        }
        std::vector<double> rf(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            rf[static_cast<std::size_t>(i)] = res[static_cast<std::size_t>(i)].to_double();
        }
        std::vector<double> delta;
        if (!solve_double(af, rf, delta)) {
            return std::nullopt;
        }
        for (int i = 0; i < n; ++i) {
            x[static_cast<std::size_t>(i)] += delta[static_cast<std::size_t>(i)];
        }
        const int shift = 12 + 2 * round;
        const int capped = shift > 30 ? 30 : shift;
        const BigInt cap(std::int64_t{1} << capped);
        std::vector<Rat> guess(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            if (!std::isfinite(x[static_cast<std::size_t>(i)])) {
                return std::nullopt;
            }
            guess[static_cast<std::size_t>(i)] = limit_denominator(rat_from_double(x[static_cast<std::size_t>(i)]), cap);
        }
        if (exact_zero(residual(a, guess, b))) {
            return guess;
        }
    }
    return std::nullopt;
}

std::optional<Solved> solve_system(const std::vector<std::vector<Rat>>& a, const std::vector<Rat>& b) {
    if (const auto refined = solve_refined(a, b)) {
        return Solved{*refined, "iterative_refinement"};
    }
    try {
        return Solved{solve_exact(a, b), "exact_lu"};
    } catch (const Singular&) {
        return std::nullopt;
    }
}

}  // namespace exact
}  // namespace qenivo
