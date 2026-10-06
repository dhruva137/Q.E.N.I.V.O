// Supernodal left-looking LDL'.
//
// Each fundamental supernode is stored as a dense trapezoid (unit lower block
// times a diagonal D). Earlier supernodes contribute by a dense rank update, the
// left-looking scheme of E. G. Ng and B. W. Peyton, SIAM J. Sci. Comput. 14 (1993).
// A pivot that is too small is shifted, which is the dynamic regularisation of
// A. Altman and J. Gondzio, Optimization Methods and Software 11 (1999), in the
// primal-dual sense of M. P. Friedlander and D. Orban, Mathematical Programming
// Computation 4 (2012). Independent branches of the supernodal elimination tree run
// on separate std::thread workers; a supernode starts only after its children finish.

#include "ipm_internal.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace qenivo {
namespace {

void expand_full(const Assembled& lower, std::vector<int>& colptr, std::vector<int>& rowidx, std::vector<double>& values) {
    const int n = lower.n;
    std::vector<std::vector<std::pair<int, double>>> cols(n);
    for (int j = 0; j < n; ++j) {
        for (int p = lower.colptr[j]; p < lower.colptr[j + 1]; ++p) {
            int r = lower.rowidx[p];
            double v = lower.values[p];
            if (r < j) {
                continue;
            }
            cols[j].push_back({r, v});
            if (r != j) {
                cols[r].push_back({j, v});
            }
        }
    }
    colptr.assign(n + 1, 0);
    rowidx.clear();
    values.clear();
    for (int j = 0; j < n; ++j) {
        auto& col = cols[j];
        std::sort(col.begin(), col.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        colptr[j] = (int)rowidx.size();
        for (size_t p = 0; p < col.size();) {
            int r = col[p].first;
            double s = 0.0;
            while (p < col.size() && col[p].first == r) {
                s += col[p].second;
                ++p;
            }
            if (s != 0.0 && s == s) {
                rowidx.push_back(r);
                values.push_back(s);
            }
        }
    }
    colptr[n] = (int)rowidx.size();
}

}  // namespace

bool Factor::run(int order_n, const int* colptr, const int* rowidx, const double* values,
                 const std::vector<int>& elimination, int spd, double reg, int threads) {
    n = order_n;
    nneg = 0;
    reg_added = 0.0;
    threads_used = 1;
    sn.clear();
    sn_parent.clear();
    if (n <= 0) {
        return true;
    }
    perm = elimination;
    if (!is_permutation(perm, n)) {
        perm.resize(n);
        for (int i = 0; i < n; ++i) {
            perm[i] = i;
        }
    }
    inv.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        inv[perm[i]] = i;
    }
    Assembled lower = assemble_lower(n, colptr, rowidx, values, perm.data());
    Assembled original = assemble_lower(n, colptr, rowidx, values, nullptr);
    expand_full(original, sp_colptr, sp_rowidx, sp_values);
    Symbolic sym = symbolic_factor(lower);
    const int ns = (int)sym.sn_start.size() - 1;
    sn.resize(ns);
    sn_parent = sym.sn_parent;
    for (int s = 0; s < ns; ++s) {
        if (sn_parent[s] <= s) {
            sn_parent[s] = -1;
        }
        Supernode& node = sn[s];
        node.c0 = sym.sn_start[s];
        node.c1 = sym.sn_start[s + 1];
        node.idx.reserve(node.c1 - node.c0 + sym.pattern[node.c0].size());
        for (int c = node.c0; c < node.c1; ++c) {
            node.idx.push_back(c);
        }
        for (int r : sym.pattern[node.c0]) {
            if (r >= node.c1) {
                node.idx.push_back(r);
            }
        }
    }
    std::vector<int> col_sn(n, -1);
    for (int s = 0; s < ns; ++s) {
        for (int c = sn[s].c0; c < sn[s].c1; ++c) {
            col_sn[c] = s;
        }
    }
    for (int u = 0; u < ns; ++u) {
        std::vector<char> hit(ns, 0);
        for (int g : sn[u].idx) {
            if (g < sn[u].c1 || g < 0 || g >= n) {
                continue;
            }
            int dest = col_sn[g];
            if (dest >= 0 && dest != u && !hit[dest]) {
                hit[dest] = 1;
                sn[dest].contrib.push_back(u);
            }
        }
    }
    const double tiny = std::max(reg, 1e-15 * lower.scale);
    auto factor_one = [&](int s) {
        Supernode& node = sn[s];
        const int w = node.c1 - node.c0;
        const int mf = (int)node.idx.size();
        std::vector<double> frontal(mf * w, 0.0);
        std::vector<int> mark(n, 0);
        std::vector<int> local(n, -1);
        for (int t = 0; t < mf; ++t) {
            mark[node.idx[t]] = 1;
            local[node.idx[t]] = t;
        }
        for (int j = node.c0; j < node.c1; ++j) {
            const int col = j - node.c0;
            for (int p = lower.colptr[j]; p < lower.colptr[j + 1]; ++p) {
                int r = lower.rowidx[p];
                if (r >= j && mark[r]) {
                    frontal[local[r] + col * mf] += lower.values[p];
                }
            }
        }
        for (int u : node.contrib) {
            const Supernode& src = sn[u];
            const int wu = src.c1 - src.c0;
            const int mfu = (int)src.idx.size();
            std::vector<int> map(mfu, -1);
            for (int t = 0; t < mfu; ++t) {
                int g = src.idx[t];
                if (g >= 0 && g < n && mark[g]) {
                    map[t] = local[g];
                }
            }
            for (int c = 0; c < wu; ++c) {
                const double dc = src.D[c];
                for (int tsrc = 0; tsrc < mfu; ++tsrc) {
                    const int ls = map[tsrc];
                    if (ls < 0 || ls >= w) {
                        continue;
                    }
                    const double lsrc = src.L[tsrc + c * mfu] * dc;
                    if (lsrc == 0.0) {
                        continue;
                    }
                    for (int tdst = 0; tdst < mfu; ++tdst) {
                        const int ld = map[tdst];
                        if (ld < ls) {
                            continue;
                        }
                        frontal[ld + ls * mf] -= lsrc * src.L[tdst + c * mfu];
                    }
                }
            }
        }
        node.L.assign(mf * w, 0.0);
        node.D.assign(w, 0.0);
        node.nneg = 0;
        node.reg_added = 0.0;
        for (int j = 0; j < w; ++j) {
            double diag = frontal[j + j * mf];
            for (int c = 0; c < j; ++c) {
                diag -= node.L[j + c * mf] * node.D[c] * node.L[j + c * mf];
            }
            if (spd) {
                if (!(diag > tiny)) {
                    node.reg_added += tiny - diag;
                    diag = tiny;
                }
            } else {
                if (!(std::fabs(diag) > tiny)) {
                    const double target = std::copysign(tiny, diag >= 0.0 ? 1.0 : -1.0);
                    node.reg_added += std::fabs(target - diag);
                    diag = target;
                }
                if (diag < 0.0) {
                    ++node.nneg;
                }
            }
            if (!(diag == diag) || diag == 0.0) {
                diag = spd ? tiny : std::copysign(tiny, 1.0);
                node.reg_added += tiny;
            }
            node.D[j] = diag;
            node.L[j + j * mf] = 1.0;
            const double inv = 1.0 / diag;
            for (int i = j + 1; i < mf; ++i) {
                double v = frontal[i + j * mf];
                for (int c = 0; c < j; ++c) {
                    v -= node.L[i + c * mf] * node.D[c] * node.L[j + c * mf];
                }
                node.L[i + j * mf] = v * inv;
            }
        }
    };

    std::vector<int> kids(ns, 0);
    for (int s = 0; s < ns; ++s) {
        if (sn_parent[s] >= 0) {
            ++kids[sn_parent[s]];
        }
    }
    int nthreads = threads;
    if (nthreads <= 0) {
        unsigned hc = std::thread::hardware_concurrency();
        nthreads = (n >= 48 && ns >= 2 && hc > 1) ? (int)std::min<unsigned>(hc, (unsigned)ns) : 1;
    }
    if (nthreads > ns) {
        nthreads = std::max(ns, 1);
    }
    if (ns <= 1) {
        nthreads = 1;
    }
    threads_used = nthreads;
    if (nthreads <= 1) {
        std::vector<int> ready;
        std::vector<int> left = kids;
        for (int s = 0; s < ns; ++s) {
            if (kids[s] == 0) {
                ready.push_back(s);
            }
        }
        for (size_t h = 0; h < ready.size(); ++h) {
            int s = ready[h];
            factor_one(s);
            int p = sn_parent[s];
            if (p >= 0 && --left[p] == 0) {
                ready.push_back(p);
            }
        }
    } else {
        std::vector<std::atomic<int>> left(ns);
        for (int s = 0; s < ns; ++s) {
            left[s].store(kids[s]);
        }
        std::mutex mu;
        std::condition_variable cv;
        std::vector<int> ready;
        for (int s = 0; s < ns; ++s) {
            if (kids[s] == 0) {
                ready.push_back(s);
            }
        }
        std::atomic<int> remaining(ns);
        auto worker = [&]() {
            for (;;) {
                int s = -1;
                {
                    std::unique_lock<std::mutex> lock(mu);
                    cv.wait(lock, [&] { return !ready.empty() || remaining.load() == 0; });
                    if (ready.empty()) {
                        return;
                    }
                    s = ready.back();
                    ready.pop_back();
                }
                factor_one(s);
                int parent = sn_parent[s];
                if (remaining.fetch_sub(1) == 1) {
                    std::lock_guard<std::mutex> lock(mu);
                    cv.notify_all();
                }
                if (parent >= 0 && left[parent].fetch_sub(1) == 1) {
                    std::lock_guard<std::mutex> lock(mu);
                    ready.push_back(parent);
                    cv.notify_one();
                }
            }
        };
        std::vector<std::thread> pool;
        pool.reserve(nthreads);
        for (int t = 0; t < nthreads; ++t) {
            pool.emplace_back(worker);
        }
        for (auto& th : pool) {
            th.join();
        }
    }
    for (const Supernode& node : sn) {
        nneg += node.nneg;
        reg_added += node.reg_added;
    }
    return true;
}

void Factor::solve(const double* b, double* x) const {
    if (n <= 0) {
        return;
    }
    std::vector<double> z(n, 0.0);
    auto once = [&](const double* rhs, double* sol) {
        for (int i = 0; i < n; ++i) {
            z[inv[i]] = rhs[i];
        }
        for (const Supernode& node : sn) {
            const int w = node.c1 - node.c0;
            const int mf = (int)node.idx.size();
            for (int j = 0; j < w; ++j) {
                const double zj = z[node.idx[j]];
                for (int i = j + 1; i < mf; ++i) {
                    z[node.idx[i]] -= node.L[i + j * mf] * zj;
                }
            }
        }
        for (const Supernode& node : sn) {
            const int w = node.c1 - node.c0;
            for (int j = 0; j < w; ++j) {
                z[node.idx[j]] /= node.D[j];
            }
        }
        for (int s = (int)sn.size() - 1; s >= 0; --s) {
            const Supernode& node = sn[s];
            const int w = node.c1 - node.c0;
            const int mf = (int)node.idx.size();
            for (int j = w - 1; j >= 0; --j) {
                double acc = z[node.idx[j]];
                for (int i = j + 1; i < mf; ++i) {
                    acc -= node.L[i + j * mf] * z[node.idx[i]];
                }
                z[node.idx[j]] = acc;
            }
        }
        for (int i = 0; i < n; ++i) {
            sol[perm[i]] = z[i];
        }
    };
    once(b, x);
    const int refine_steps = 2;
    std::vector<double> residual(n, 0.0);
    std::vector<double> step(n, 0.0);
    for (int it = 0; it < refine_steps; ++it) {
        std::fill(residual.begin(), residual.end(), 0.0);
        for (int j = 0; j < n; ++j) {
            for (int p = sp_colptr[j]; p < sp_colptr[j + 1]; ++p) {
                residual[sp_rowidx[p]] += sp_values[p] * x[j];
            }
        }
        double nr = 0.0;
        for (int i = 0; i < n; ++i) {
            residual[i] = b[i] - residual[i];
            nr += residual[i] * residual[i];
        }
        if (!(nr > 0.0)) {
            break;
        }
        once(residual.data(), step.data());
        for (int i = 0; i < n; ++i) {
            x[i] += step[i];
        }
    }
}

}  // namespace qenivo
