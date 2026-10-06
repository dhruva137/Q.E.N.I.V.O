// Tiny assert harness for ordering, the elimination tree and numeric LDL'.
// No GoogleTest. ctest runs this binary.

#include "ipm_internal.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_fails = 0;

static void check(bool cond, const char* text, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL %s:%d %s\n", "test_factor.cpp", line, text);
        ++g_fails;
    }
}

#define CHECK(cond) check((cond), #cond, __LINE__)

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s = s * 6364136223846793005ull + 1ull;
        return s;
    }
    int mod(int n) { return (int)(next() % (uint64_t)n); }
    double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
};

struct Csc {
    int n = 0;
    std::vector<int> colptr;
    std::vector<int> rowidx;
    std::vector<double> values;
};

Csc from_dense(const std::vector<double>& a, int n) {
    Csc m;
    m.n = n;
    m.colptr.push_back(0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            double v = a[i + j * n];
            if (v != 0.0) {
                m.rowidx.push_back(i);
                m.values.push_back(v);
            }
        }
        m.colptr.push_back((int)m.rowidx.size());
    }
    return m;
}

std::vector<double> matvec(const Csc& m, const std::vector<double>& x) {
    std::vector<double> y(m.n, 0.0);
    for (int j = 0; j < m.n; ++j) {
        for (int p = m.colptr[j]; p < m.colptr[j + 1]; ++p) {
            y[m.rowidx[p]] += m.values[p] * x[j];
        }
    }
    return y;
}

double residual(const Csc& m, const std::vector<double>& x, const std::vector<double>& b) {
    std::vector<double> ax = matvec(m, x);
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < m.n; ++i) {
        double r = ax[i] - b[i];
        num += r * r;
        den += b[i] * b[i];
    }
    return std::sqrt(num) / std::sqrt(den + 1e-300);
}

Csc random_spd(int n, uint64_t seed) {
    Rng rng(seed);
    std::vector<double> a(n * n, 0.0);
    for (int k = 0; k < n; ++k) {
        std::vector<double> col(n, 0.0);
        col[k] = 0.5 + rng.uniform();
        for (int t = 0; t < 3; ++t) {
            col[rng.mod(n)] += 0.3 * (rng.uniform() - 0.5);
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                a[i + j * n] += col[i] * col[j];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        a[i + i * n] += 1.0;
    }
    return from_dense(a, n);
}

Csc block_spd(int n, uint64_t seed) {
    Csc left = random_spd(n / 2, seed);
    Csc right = random_spd(n - n / 2, seed + 1);
    std::vector<double> a(n * n, 0.0);
    auto paste = [&](const Csc& src, int off) {
        for (int j = 0; j < src.n; ++j) {
            for (int p = src.colptr[j]; p < src.colptr[j + 1]; ++p) {
                a[src.rowidx[p] + off + (j + off) * n] = src.values[p];
            }
        }
    };
    paste(left, 0);
    paste(right, n / 2);
    return from_dense(a, n);
}

Csc small_kkt(int q, int m, uint64_t seed) {
    Rng rng(seed);
    const int n = q + m;
    std::vector<double> a(n * n, 0.0);
    for (int j = 0; j < q; ++j) {
        a[j + j * n] = 2.0;
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < q; ++j) {
            if (rng.mod(3) == 0) {
                double v = 0.2 * (rng.uniform() - 0.5);
                a[(q + i) + j * n] = v;
                a[j + (q + i) * n] = v;
            }
        }
        a[(q + i) + (q + i) * n] = -0.5;
    }
    return from_dense(a, n);
}

void check_order_and_factor(const Csc& m, int method, int spd, int threads, const char* label) {
    std::vector<int> perm(m.n);
    int rc = nr_ipm_order(m.n, m.colptr.data(), m.rowidx.data(), method, perm.data());
    CHECK(rc == 0);
    CHECK(qenivo::is_permutation(perm, m.n));
    std::vector<int> parent(m.n), colcount(m.n), sn_id(m.n);
    int ns = -1;
    rc = nr_ipm_symbolic(m.n, m.colptr.data(), m.rowidx.data(), perm.data(), parent.data(), colcount.data(), sn_id.data(),
                         &ns);
    CHECK(rc == 0);
    CHECK(ns >= 1);
    for (int j = 0; j < m.n; ++j) {
        CHECK(colcount[j] >= 1);
        CHECK(colcount[j] <= m.n - j);
        CHECK(parent[j] == -1 || (parent[j] > j && parent[j] < m.n));
        CHECK(sn_id[j] >= 0 && sn_id[j] < ns);
    }
    int nneg = -1;
    double reg = -1.0;
    void* factor = nr_ipm_factor(m.n, m.colptr.data(), m.rowidx.data(), m.values.data(), perm.data(), spd, 0.0, threads,
                                 &nneg, &reg);
    CHECK(factor != nullptr);
    std::vector<double> b(m.n), x(m.n, 0.0);
    for (int i = 0; i < m.n; ++i) {
        b[i] = 0.2 * (i + 1) - 0.5 * ((i * 3) % 5);
    }
    CHECK(nr_ipm_solve(factor, b.data(), x.data()) == 0);
    double rel = residual(m, x, b);
    if (!(rel <= 1e-12)) {
        std::fprintf(stderr, "residual %s method %d threads %d = %.3e (nneg %d reg %.3e)\n", label, method, threads, rel,
                     nneg, reg);
    }
    CHECK(rel <= 1e-12);
    nr_ipm_free(factor);
}

}  // namespace

int main() {
    Csc eye = from_dense({2.0}, 1);
    check_order_and_factor(eye, 0, 1, 1, "1x1");

    Csc spd = random_spd(24, 11);
    check_order_and_factor(spd, 0, 1, 1, "spd-amd");
    check_order_and_factor(spd, 1, 1, 1, "spd-nd");

    Csc blocks = block_spd(36, 19);
    check_order_and_factor(blocks, 0, 1, 2, "spd-threads");

    Csc kkt = small_kkt(8, 4, 23);
    check_order_and_factor(kkt, 0, 0, 1, "kkt");

    if (g_fails == 0) {
        std::printf("ok\n");
        return 0;
    }
    std::fprintf(stderr, "%d checks failed\n", g_fails);
    return 1;
}
