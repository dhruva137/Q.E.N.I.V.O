// QENIVO native simplex core (C++17, no dependencies).
//
// Bounded-variable revised simplex on the standard form used by engines/simplex.py:
//     min [c 0]'[x s]   s.t.  [A -I][x s] = 0,   [lx lc] <= [x s] <= [ux uc]
// Components
//   * sparse LU of the basis: left-looking, threshold partial pivoting (0.1), slack columns first,
//     structural columns by increasing length; L and U kept by columns and by rows, so FTRAN and
//     BTRAN both run in scatter form and skip the zeros of sparse right-hand sides
//   * product-form updates with sparse eta columns between refactorisations
//   * dual simplex (the main algorithm, cold or warm): dual steepest-edge pricing, bound-flipping
//     ratio test with Harris tolerances, row-wise PRICE for sparse pivot rows, incremental reduced
//     costs and basic values (recomputed at every refactorisation), cost perturbation against
//     degeneracy and cost shifting for small dual infeasibilities; dual phase 1 on the auxiliary
//     box problem (free -> [-1000,1000], lower -> [0,1], upper -> [-1,0], boxed -> [0,0])
//   * primal simplex: phase 2 with Devex pricing and incremental reduced costs (cleanup once the
//     perturbation is removed, and primal feasible warm starts); composite phase 1 as the fallback
//     (dual infeasible models: unboundedness is decided there); Harris ratio test with bound flips;
//     Bland's rule after a run of degenerate pivots
// Interface: extern "C" nr_simplex(...) with status codes, see engines/native_simplex.py.
// Every answer is re-checked in Python by the float64 KKT test on the original model.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

const double INF = std::numeric_limits<double>::infinity();
enum { BASIC = 0, AT_LO = 1, AT_UP = 2, FREE_NB = 3 };
enum { S_OPTIMAL = 0, S_INFEASIBLE = 1, S_UNBOUNDED = 2, S_ITER = 3, S_TIME = 4, S_NUM = 5 };
enum { P_FACTOR, P_FTRAN, P_BTRAN, P_PRICE, P_CHUZC, P_CHUZR, P_UPDATE, P_OTHER, P_N };

double g_prof[P_N];
long g_count[P_N];
using Clock = std::chrono::steady_clock;

struct Tic {
    int k;
    Clock::time_point t;
    explicit Tic(int k_) : k(k_), t(Clock::now()) {}
    ~Tic() { g_prof[k] += std::chrono::duration<double>(Clock::now() - t).count(); g_count[k]++; }
};

struct Factor {
    std::vector<int> prow, rowpos, qcol;                 // pivot row of step k, step of row i, basis pos of k
    std::vector<double> diag;
    std::vector<int> Ls, Li; std::vector<double> Lv;     // L by step: (row, multiplier)
    std::vector<int> LTs, LTi; std::vector<double> LTv;  // L by pivot-row step: (earlier step, multiplier)
    std::vector<int> Us, Ui; std::vector<double> Uv;     // U column k: (step p < k, value)
    std::vector<int> UTs, UTi; std::vector<double> UTv;  // U row p: (step k > p, value)
    std::vector<int> Es, Ei, Er; std::vector<double> Ev, Ep;  // etas: position r, pivot, (pos, alpha)
    long lu_nnz = 0;
    int netas() const { return (int)Er.size(); }
};

struct Simplex {
    int m = 0, n = 0, N = 0;
    const int *Ap = nullptr, *Ai = nullptr;
    const double* Ax = nullptr;
    std::vector<int> Rp, Rj;                    // A by rows (row-wise PRICE)
    std::vector<double> Rx;
    std::vector<double> cost, lo, hi, x, d;
    std::vector<int> st, basis;
    std::vector<double> dse;                    // dual steepest-edge weights (by basis position)
    Factor F;
    double ptol = 1e-9, dtol = 1e-9;
    long iters = 0, max_iter = 1000000;
    double time_limit = 1e300;
    Clock::time_point t0;
    int refactor_every = 100;
    int refactors = 0;
    std::vector<double> ybuf;
    uint64_t rng = 0x9E3779B97F4A7C15ull;

    double rand01() {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        return (rng >> 11) * (1.0 / 9007199254740992.0);
    }

    void build_rows() {
        Rp.assign(m + 1, 0);
        for (int k = 0; k < Ap[n]; ++k) Rp[Ai[k] + 1]++;
        for (int i = 0; i < m; ++i) Rp[i + 1] += Rp[i];
        Rj.resize(Ap[n]); Rx.resize(Ap[n]);
        std::vector<int> pos(Rp.begin(), Rp.end() - 1);
        for (int j = 0; j < n; ++j)
            for (int k = Ap[j]; k < Ap[j + 1]; ++k) { int p = pos[Ai[k]]++; Rj[p] = j; Rx[p] = Ax[k]; }
    }

    void add_col(int j, double s, std::vector<double>& v) const {
        if (j < n) { for (int k = Ap[j]; k < Ap[j + 1]; ++k) v[Ai[k]] += s * Ax[k]; }
        else v[j - n] -= s;
    }
    double dot_col(int j, const std::vector<double>& v) const {
        if (j < n) {
            double s = 0;
            for (int k = Ap[j]; k < Ap[j + 1]; ++k) s += Ax[k] * v[Ai[k]];
            return s;
        }
        return -v[j - n];
    }
    int col_len(int j) const { return j < n ? Ap[j + 1] - Ap[j] : 1; }
    bool boxed(int j) const { return lo[j] > -INF && hi[j] < INF; }
    bool timed_out() const { return std::chrono::duration<double>(Clock::now() - t0).count() > time_limit; }

    // ---------------------------------------------------------------- factorisation
    bool factorize() {
        Tic tic(P_FACTOR);
        const int M = m;
        Factor f;
        f.prow.assign(M, -1); f.rowpos.assign(M, -1); f.qcol.resize(M); f.diag.assign(M, 0.0);
        std::vector<std::vector<std::pair<int, double>>> L(M), U(M);
        std::vector<int> order(M);
        for (int k = 0; k < M; ++k) order[k] = k;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return col_len(basis[a]) < col_len(basis[b]); });
        std::vector<int> rowcount(M, 0);                          // row lengths of B (Markowitz tie-break)
        for (int pos = 0; pos < M; ++pos) {
            int j = basis[pos];
            if (j < n) { for (int q = Ap[j]; q < Ap[j + 1]; ++q) rowcount[Ai[q]]++; }
            else rowcount[j - n]++;
        }
        std::vector<double> w(M, 0.0);
        std::vector<int> nzrows, steps;
        std::vector<char> mark(M, 0);
        for (int k = 0; k < M; ++k) {
            int pos = order[k];
            int j = basis[pos];
            f.qcol[k] = pos;
            nzrows.clear();
            auto touch = [&](int r) { if (!mark[r]) { mark[r] = 1; nzrows.push_back(r); } };
            if (j < n) { for (int q = Ap[j]; q < Ap[j + 1]; ++q) { w[Ai[q]] += Ax[q]; touch(Ai[q]); } }
            else { w[j - n] -= 1.0; touch(j - n); }
            // left-looking: eliminate with every earlier pivot whose row is touched, in pivot order
            steps.clear();
            for (int r : nzrows) if (f.rowpos[r] >= 0) steps.push_back(f.rowpos[r]);
            std::sort(steps.begin(), steps.end());
            for (size_t a = 0; a < steps.size(); ++a) {
                int p = steps[a];
                if (a && steps[a] == steps[a - 1]) continue;
                double coef = w[f.prow[p]];
                if (coef == 0.0) continue;
                U[k].push_back({p, coef});
                for (auto& e : L[p]) {
                    int r = e.first;
                    w[r] -= e.second * coef;
                    if (!mark[r]) {
                        mark[r] = 1; nzrows.push_back(r);
                        if (f.rowpos[r] >= 0 && f.rowpos[r] > p)
                            steps.insert(std::upper_bound(steps.begin() + a + 1, steps.end(), f.rowpos[r]), f.rowpos[r]);
                    }
                }
            }
            double big = 0.0;
            for (int r : nzrows) if (f.rowpos[r] < 0) big = std::max(big, std::fabs(w[r]));
            if (big < 1e-11) {                                    // singular basis
                for (int r : nzrows) { w[r] = 0.0; mark[r] = 0; }
                return false;
            }
            int piv = -1, best_len = 1 << 30;
            for (int r : nzrows) {                                // threshold 0.1, then the sparsest row
                if (f.rowpos[r] >= 0 || std::fabs(w[r]) < 0.1 * big) continue;
                int len = rowcount[r];
                if (len < best_len || (len == best_len && std::fabs(w[r]) > std::fabs(w[piv]))) { best_len = len; piv = r; }
            }
            f.prow[k] = piv;
            f.rowpos[piv] = k;
            const double pv = w[piv];
            f.diag[k] = pv;
            for (int r : nzrows) {
                if (f.rowpos[r] < 0 && w[r] != 0.0) L[k].push_back({r, w[r] / pv});
                w[r] = 0.0; mark[r] = 0;
            }
        }
        // flatten, and build the transposed copies
        f.Ls.assign(M + 1, 0); f.Us.assign(M + 1, 0); f.LTs.assign(M + 1, 0); f.UTs.assign(M + 1, 0);
        for (int k = 0; k < M; ++k) {
            f.Ls[k + 1] = f.Ls[k] + (int)L[k].size();
            f.Us[k + 1] = f.Us[k] + (int)U[k].size();
            for (auto& e : L[k]) f.LTs[f.rowpos[e.first] + 1]++;
            for (auto& e : U[k]) f.UTs[e.first + 1]++;
        }
        for (int k = 0; k < M; ++k) { f.LTs[k + 1] += f.LTs[k]; f.UTs[k + 1] += f.UTs[k]; }
        f.Li.resize(f.Ls[M]); f.Lv.resize(f.Ls[M]); f.LTi.resize(f.Ls[M]); f.LTv.resize(f.Ls[M]);
        f.Ui.resize(f.Us[M]); f.Uv.resize(f.Us[M]); f.UTi.resize(f.Us[M]); f.UTv.resize(f.Us[M]);
        std::vector<int> lp(f.LTs.begin(), f.LTs.end() - 1), up(f.UTs.begin(), f.UTs.end() - 1);
        for (int k = 0; k < M; ++k) {
            int a = f.Ls[k];
            for (auto& e : L[k]) {
                f.Li[a] = e.first; f.Lv[a++] = e.second;
                int t = lp[f.rowpos[e.first]]++;
                f.LTi[t] = k; f.LTv[t] = e.second;
            }
            a = f.Us[k];
            for (auto& e : U[k]) {
                f.Ui[a] = e.first; f.Uv[a++] = e.second;
                int t = up[e.first]++;
                f.UTi[t] = k; f.UTv[t] = e.second;
            }
        }
        f.Es.assign(1, 0);
        f.lu_nnz = f.Ls[M] + f.Us[M] + M;
        F = std::move(f);
        ++refactors;
        return true;
    }

    // B z = b: v by row in, by basis position out
    void ftran(std::vector<double>& v) {
        Tic tic(P_FTRAN);
        const int M = m;
        std::vector<double>& y = ybuf;
        for (int p = 0; p < M; ++p) {
            double t = v[F.prow[p]];
            if (t == 0.0) continue;
            for (int a = F.Ls[p]; a < F.Ls[p + 1]; ++a) v[F.Li[a]] -= F.Lv[a] * t;
        }
        for (int p = 0; p < M; ++p) y[p] = v[F.prow[p]];
        for (int k = M - 1; k >= 0; --k) {
            double t = y[k];
            if (t == 0.0) { v[F.qcol[k]] = 0.0; continue; }
            t /= F.diag[k];
            v[F.qcol[k]] = t;
            for (int a = F.Us[k]; a < F.Us[k + 1]; ++a) y[F.Ui[a]] -= F.Uv[a] * t;
        }
        const int ne = F.netas();
        for (int e = 0; e < ne; ++e) {
            int r = F.Er[e];
            double zr = v[r];
            if (zr == 0.0) continue;
            zr /= F.Ep[e];
            v[r] = zr;
            for (int a = F.Es[e]; a < F.Es[e + 1]; ++a) v[F.Ei[a]] -= F.Ev[a] * zr;
        }
    }

    // B' pi = d: v by basis position in, by row out
    void btran(std::vector<double>& v) {
        Tic tic(P_BTRAN);
        const int M = m;
        for (int e = F.netas() - 1; e >= 0; --e) {
            int r = F.Er[e];
            double s = v[r];
            for (int a = F.Es[e]; a < F.Es[e + 1]; ++a) s -= F.Ev[a] * v[F.Ei[a]];
            v[r] = s / F.Ep[e];
        }
        std::vector<double>& y = ybuf;
        for (int k = 0; k < M; ++k) y[k] = v[F.qcol[k]];
        for (int k = 0; k < M; ++k) {
            double t = y[k];
            if (t == 0.0) continue;
            t /= F.diag[k];
            y[k] = t;
            for (int a = F.UTs[k]; a < F.UTs[k + 1]; ++a) y[F.UTi[a]] -= F.UTv[a] * t;
        }
        for (int k = M - 1; k >= 0; --k) {
            double t = y[k];
            v[F.prow[k]] = t;
            if (t == 0.0) continue;
            for (int a = F.LTs[k]; a < F.LTs[k + 1]; ++a) y[F.LTi[a]] -= F.LTv[a] * t;
        }
    }

    // basis change at position r with alpha = B^-1 a_q; false: refactorise instead (caller does)
    bool push_eta(int r, const std::vector<double>& alpha, const std::vector<int>& nz) {
        if (F.netas() >= refactor_every || (long)F.Ei.size() > 3 * F.lu_nnz + 10L * m) return false;
        for (int i : nz) {
            if (i == r || alpha[i] == 0.0) continue;
            F.Ei.push_back(i); F.Ev.push_back(alpha[i]);
        }
        F.Er.push_back(r); F.Ep.push_back(alpha[r]); F.Es.push_back((int)F.Ei.size());
        return true;
    }

    double nb_value(int j) const {
        if (st[j] == AT_LO) return lo[j];
        if (st[j] == AT_UP) return hi[j];
        return 0.0;
    }

    void compute_xb() {
        std::vector<double> rhs(m, 0.0);
        for (int j = 0; j < N; ++j) {
            if (st[j] == BASIC) continue;
            x[j] = nb_value(j);
            if (x[j] != 0.0) add_col(j, -x[j], rhs);
        }
        ftran(rhs);
        for (int k = 0; k < m; ++k) x[basis[k]] = rhs[k];
    }

    void compute_pi(const std::vector<double>& c, std::vector<double>& pi) {
        pi.assign(m, 0.0);
        for (int k = 0; k < m; ++k) pi[k] = c[basis[k]];
        btran(pi);
    }

    void compute_d() {
        std::vector<double> pi;
        compute_pi(cost, pi);
        for (int j = 0; j < N; ++j) d[j] = st[j] == BASIC ? 0.0 : cost[j] - dot_col(j, pi);
    }

    bool primal_feasible(double tol) const {
        for (int k = 0; k < m; ++k) {
            int j = basis[k];
            if (x[j] < lo[j] - tol || x[j] > hi[j] + tol) return false;
        }
        return true;
    }

    // dual infeasibility of nonbasic j (> 0 when infeasible)
    double dual_infeas(int j) const {
        if (st[j] == BASIC || lo[j] == hi[j]) return 0.0;
        if (st[j] == AT_LO) return -d[j];
        if (st[j] == AT_UP) return d[j];
        return std::fabs(d[j]);
    }

    int count_dual_infeas() const {
        int c = 0;
        for (int j = 0; j < N; ++j) if (dual_infeas(j) > dtol) ++c;
        return c;
    }

    // restore dual feasibility after a recomputation: flip boxed columns, shift the cost of others
    void correct_duals() {
        bool flipped = false;
        for (int j = 0; j < N; ++j) {
            if (dual_infeas(j) <= dtol) continue;
            if (boxed(j)) { st[j] = (st[j] == AT_LO) ? AT_UP : AT_LO; flipped = true; }
            else { cost[j] -= d[j]; d[j] = 0.0; }
        }
        if (flipped) compute_xb();
    }

    void recompute() { compute_xb(); compute_d(); }

    // pivot row: arow[j] = rho' a_j for nonbasic j (rho by row); anz lists the touched columns
    void price(const std::vector<double>& rho, const std::vector<int>& rnz, std::vector<double>& arow,
               std::vector<int>& anz, std::vector<char>& mark) {
        Tic tic(P_PRICE);
        anz.clear();
        long work = 0;
        for (int i : rnz) work += Rp[i + 1] - Rp[i];
        if (work < Ap[n] / 3) {                                   // row-wise
            for (int i : rnz) {
                double ri = rho[i];
                for (int k = Rp[i]; k < Rp[i + 1]; ++k) {
                    int j = Rj[k];
                    if (!mark[j]) { mark[j] = 1; anz.push_back(j); }
                    arow[j] += ri * Rx[k];
                }
            }
        } else {                                                  // column-wise
            for (int j = 0; j < n; ++j) {
                if (st[j] == BASIC) continue;
                double v = 0;
                for (int k = Ap[j]; k < Ap[j + 1]; ++k) v += Ax[k] * rho[Ai[k]];
                if (v != 0.0) { arow[j] = v; mark[j] = 1; anz.push_back(j); }
            }
        }
        for (int i : rnz) {
            int j = n + i;
            if (st[j] == BASIC) continue;
            arow[j] = -rho[i]; mark[j] = 1; anz.push_back(j);
        }
    }
    static void clear_row(std::vector<double>& arow, std::vector<int>& anz, std::vector<char>& mark) {
        for (int j : anz) { arow[j] = 0.0; mark[j] = 0; }
        anz.clear();
    }
    void nzlist(const std::vector<double>& v, std::vector<int>& nz) const {
        nz.clear();
        for (int i = 0; i < m; ++i) if (v[i] != 0.0) nz.push_back(i);
    }

    // ---------------------------------------------------------------- dual simplex
    // Needs a factorised basis with x and d computed and dual feasible (up to dtol); maintains them.
    int dual() {
        std::vector<double> rho(m, 0.0), tau(m, 0.0), alpha(m, 0.0), fl(m, 0.0), arow(N, 0.0);
        std::vector<int> rnz, anz, alnz, flips;
        std::vector<char> mark(N, 0);
        struct Cand { int j; double a, ratio, relax; };
        std::vector<Cand> cand;
        std::vector<double> sufmin;
        int retry = 0;
        for (;;) {
            if (iters >= max_iter) return S_ITER;
            if ((iters & 15) == 0 && timed_out()) return S_TIME;
            int r = -1;
            {
                Tic tic(P_CHUZR);
                double best = 0;
                for (int k = 0; k < m; ++k) {
                    int j = basis[k];
                    double v = x[j], inf;
                    if (v < lo[j] - ptol) inf = lo[j] - v;
                    else if (v > hi[j] + ptol) inf = v - hi[j];
                    else continue;
                    double sc = inf * inf / dse[k];
                    if (sc > best) { best = sc; r = k; }
                }
            }
            if (r < 0) return S_OPTIMAL;
            const int jr = basis[r];
            const bool to_lo = x[jr] < lo[jr];
            const double bound = to_lo ? lo[jr] : hi[jr];
            const double s = to_lo ? -1.0 : 1.0;                   // sign of the dual step
            std::fill(rho.begin(), rho.end(), 0.0);
            rho[r] = 1.0;
            btran(rho);
            nzlist(rho, rnz);
            double rn2 = 0;
            for (int i : rnz) rn2 += rho[i] * rho[i];
            dse[r] = rn2;                                         // exact weight of the pivot row
            price(rho, rnz, arow, anz, mark);
            // bound-flipping ratio test with Harris tolerances
            int q = -1;
            flips.clear();
            {
                Tic tic(P_CHUZC);
                cand.clear();
                for (int j : anz) {
                    if (st[j] == BASIC || lo[j] == hi[j]) continue;
                    double a = s * arow[j];
                    if (std::fabs(a) <= 1e-9) continue;
                    double dd;
                    if (st[j] == AT_LO) { if (a <= 0) continue; dd = d[j]; }
                    else if (st[j] == AT_UP) { if (a >= 0) continue; dd = -d[j]; }
                    else dd = 0.0;
                    dd = std::max(dd, 0.0);
                    double aa = std::fabs(a);
                    cand.push_back({j, aa, dd / aa, (dd + dtol) / aa});
                }
                std::sort(cand.begin(), cand.end(), [](const Cand& u, const Cand& v) { return u.ratio < v.ratio; });
                const int K = (int)cand.size();
                sufmin.assign(K + 1, INF);
                for (int k = K - 1; k >= 0; --k) sufmin[k] = std::min(sufmin[k + 1], cand[k].relax);
                double slope = std::fabs(x[jr] - bound);
                int start = 0;
                while (start < K) {
                    double tmax = sufmin[start];
                    int end = start;
                    double tot = 0;
                    while (end < K && cand[end].ratio <= tmax) {
                        int j = cand[end].j;
                        tot += boxed(j) ? cand[end].a * (hi[j] - lo[j]) : INF;
                        ++end;
                    }
                    if (tot < slope) {                                   // pass the whole group: bound flips
                        for (int k = start; k < end; ++k) flips.push_back(cand[k].j);
                        slope -= tot;
                        start = end;
                        continue;
                    }
                    double bestg = -1;
                    for (int k = start; k < end; ++k) if (cand[k].a > bestg) { bestg = cand[k].a; q = cand[k].j; }
                    break;
                }
            }
            if (q < 0) {                                                  // dual ray: row r cannot be repaired
                clear_row(arow, anz, mark);
                if (F.netas() > 0 && retry < 2) {
                    ++retry;
                    if (!factorize()) return S_NUM;
                    recompute(); correct_duals();
                    continue;
                }
                return S_INFEASIBLE;
            }
            std::fill(alpha.begin(), alpha.end(), 0.0);
            add_col(q, 1.0, alpha);
            ftran(alpha);
            const double arq = alpha[r];
            if (std::fabs(arq - arow[q]) > 1e-8 * (1.0 + std::fabs(arq)) || std::fabs(arq) < 1e-11) {
                if (F.netas() > 0 && retry < 3) {                        // numerical trouble: refactorise
                    clear_row(arow, anz, mark);
                    ++retry;
                    if (!factorize()) return S_NUM;
                    recompute(); correct_duals();
                    continue;
                }
                if (std::fabs(arq) < 1e-11) { clear_row(arow, anz, mark); return S_NUM; }
            }
            retry = 0;
            ++iters;
            Tic tic(P_UPDATE);
            nzlist(alpha, alnz);
            double td = d[q] / arq;
            if (s * td < 0) { cost[q] -= d[q]; d[q] = 0.0; td = 0.0; }     // shift: no step backwards
            tau = rho;                                                    // DSE: tau = B^-1 rho
            ftran(tau);
            if (!flips.empty()) {                                         // bound flips
                std::fill(fl.begin(), fl.end(), 0.0);
                for (int j : flips) {
                    double nv = (st[j] == AT_LO) ? hi[j] : lo[j];
                    add_col(j, nv - x[j], fl);
                    x[j] = nv;
                    st[j] = (st[j] == AT_LO) ? AT_UP : AT_LO;
                }
                ftran(fl);
                for (int k = 0; k < m; ++k) if (fl[k] != 0.0) x[basis[k]] -= fl[k];
            }
            const double tp = (x[jr] - bound) / arq;                      // primal step
            for (int i : alnz) x[basis[i]] -= tp * alpha[i];
            x[q] += tp;
            x[jr] = bound;
            for (int j : anz) if (st[j] != BASIC) d[j] -= td * arow[j];   // dual step
            d[q] = 0.0;
            d[jr] = -td;
            for (int j : anz) {                                           // cost shifts for small drift
                if (j == q || st[j] == BASIC) continue;
                if (dual_infeas(j) > dtol) { cost[j] -= d[j]; d[j] = 0.0; }
            }
            clear_row(arow, anz, mark);
            {                                                             // dual steepest-edge update
                const double wr = dse[r];
                for (int i : alnz) {
                    if (i == r) continue;
                    double ratio = alpha[i] / arq;
                    double w = dse[i] + ratio * (ratio * wr - 2.0 * tau[i]);
                    dse[i] = std::max(w, 1e-4);
                }
                dse[r] = std::max(wr / (arq * arq), 1e-4);
            }
            basis[r] = q;
            st[q] = BASIC;
            st[jr] = to_lo ? AT_LO : AT_UP;
            if (!push_eta(r, alpha, alnz)) {
                if (!factorize()) return S_NUM;
                recompute(); correct_duals();
            }
        }
    }

    // ---------------------------------------------------------------- primal simplex
    // Harris two-pass ratio test on alpha for direction dir. Returns the row (-1: none).
    int primal_ratio(const std::vector<double>& alpha, const std::vector<int>& alnz, int q, double dir,
                     bool phase1, double& tstep, double& tflip, bool& unbounded) {
        tflip = boxed(q) ? hi[q] - lo[q] : INF;
        double tmax = tflip;
        auto bounds = [&](int j, double& l, double& u) {
            l = lo[j]; u = hi[j];
            if (phase1) {                          // an infeasible basic stops where it turns feasible
                if (x[j] < lo[j] - ptol) { l = -INF; u = lo[j]; }
                if (x[j] > hi[j] + ptol) { u = INF; l = hi[j]; }
            }
        };
        for (int k : alnz) {
            double g = dir * alpha[k];
            if (std::fabs(g) <= 1e-9) continue;
            int j = basis[k];
            double l, u;
            bounds(j, l, u);
            double t = g > 0 ? (l > -INF ? (x[j] - l + ptol) / g : INF) : (u < INF ? (u - x[j] + ptol) / (-g) : INF);
            tmax = std::min(tmax, t);
        }
        unbounded = (tmax == INF);
        if (unbounded) return -1;
        int r = -1;
        double bestg = 0;
        tstep = tflip;
        for (int k : alnz) {                                    // pass 2: largest |alpha|
            double g = dir * alpha[k];
            if (std::fabs(g) <= 1e-9) continue;
            int j = basis[k];
            double l, u;
            bounds(j, l, u);
            double t = g > 0 ? (l > -INF ? (x[j] - l) / g : INF) : (u < INF ? (u - x[j]) / (-g) : INF);
            if (t <= tmax && std::fabs(g) > bestg) { bestg = std::fabs(g); r = k; tstep = std::max(t, 0.0); }
        }
        return r;
    }

    // phase 1: composite (costs = infeasibility gradient, reduced costs recomputed every iteration)
    // phase 2: reduced costs updated from the pivot row, recomputed at refactorisation
    int primal(bool phase1) {
        std::vector<double> alpha(m, 0.0), rho(m, 0.0), arow(N, 0.0), cst;
        std::vector<int> alnz, rnz, anz;
        std::vector<char> mark(N, 0);
        std::vector<double> dvx(N, 1.0);                                    // Devex reference weights
        std::vector<double> saved_cost;
        if (phase1) { saved_cost = cost; cst.assign(N, 0.0); }
        auto done = [&](int code) { if (phase1) cost = saved_cost; return code; };
        int degenerate = 0, retry = 0;
        if (!phase1) compute_d();
        for (;;) {
            if (iters >= max_iter) return done(S_ITER);
            if ((iters & 15) == 0 && timed_out()) return done(S_TIME);
            if (phase1) {
                double infeas = 0;
                std::fill(cst.begin(), cst.end(), 0.0);
                for (int k = 0; k < m; ++k) {
                    int j = basis[k];
                    if (x[j] < lo[j] - ptol) { cst[j] = -1.0; infeas += lo[j] - x[j]; }
                    else if (x[j] > hi[j] + ptol) { cst[j] = 1.0; infeas += x[j] - hi[j]; }
                }
                if (infeas <= ptol * (1.0 + m)) return done(S_OPTIMAL);     // feasible
                cost.swap(cst);
                compute_d();
                cost.swap(cst);
            }
            int q = -1;
            {
                Tic tic(P_CHUZC);
                double best = 0;
                bool bland = degenerate > 50;
                for (int j = 0; j < N; ++j) {
                    if (dual_infeas(j) <= dtol) continue;
                    if (bland) { q = j; break; }
                    double sc = d[j] * d[j] / dvx[j];
                    if (sc > best) { best = sc; q = j; }
                }
            }
            if (q < 0) return done(S_OPTIMAL);
            const double dir = d[q] < 0 ? 1.0 : -1.0;
            std::fill(alpha.begin(), alpha.end(), 0.0);
            add_col(q, 1.0, alpha);
            ftran(alpha);
            nzlist(alpha, alnz);
            double tstep = 0, tflip = INF;
            bool unb = false;
            int r = primal_ratio(alpha, alnz, q, dir, phase1, tstep, tflip, unb);
            if (unb) {
                if (F.netas() > 0 && retry < 2) {
                    ++retry;
                    if (!factorize()) return done(S_NUM);
                    compute_xb(); if (!phase1) compute_d();
                    continue;
                }
                return done(phase1 ? S_NUM : S_UNBOUNDED);
            }
            if (r < 0 || tflip <= tstep) {                                   // bound flip of the entering column
                ++iters;
                for (int k : alnz) x[basis[k]] -= dir * tflip * alpha[k];
                st[q] = (st[q] == AT_LO) ? AT_UP : AT_LO;
                x[q] = nb_value(q);
                degenerate = 0;
                continue;
            }
            const double arq = alpha[r];
            std::fill(rho.begin(), rho.end(), 0.0);
            rho[r] = 1.0;
            btran(rho);
            nzlist(rho, rnz);
            price(rho, rnz, arow, anz, mark);
            if (std::fabs(arow[q] - arq) > 1e-8 * (1.0 + std::fabs(arq)) && F.netas() > 0 && retry < 3) {
                clear_row(arow, anz, mark);
                ++retry;
                if (!factorize()) return done(S_NUM);
                compute_xb(); if (!phase1) compute_d();
                continue;
            }
            retry = 0;
            ++iters;
            if (!phase1) {                                                   // incremental reduced costs
                const double td = d[q] / arq;
                for (int j : anz) if (st[j] != BASIC) d[j] -= td * arow[j];
                d[basis[r]] = -td;
                d[q] = 0.0;
            }
            {                                                                // Devex weights (pivot row)
                const double wq = dvx[q];
                double wmax = 0;
                for (int j : anz) {
                    if (st[j] == BASIC || j == q) continue;
                    double a = arow[j] / arq;
                    double w = a * a * wq;
                    if (w > dvx[j]) dvx[j] = w;
                    wmax = std::max(wmax, dvx[j]);
                }
                dvx[basis[r]] = std::max(wq / (arq * arq), 1.0);
                if (wmax > 1e8) std::fill(dvx.begin(), dvx.end(), 1.0);      // reset the reference framework
            }
            clear_row(arow, anz, mark);
            for (int k : alnz) x[basis[k]] -= dir * tstep * alpha[k];
            const int leave = basis[r];
            x[q] = nb_value(q) + dir * tstep;
            {   // leaving variable goes to the bound it reached (decided by position: phase 1 may
                // reach the near bound from outside)
                double xv = x[leave];
                double dl = lo[leave] > -INF ? std::fabs(xv - lo[leave]) : INF;
                double du = hi[leave] < INF ? std::fabs(xv - hi[leave]) : INF;
                st[leave] = (dl <= du) ? AT_LO : AT_UP;
            }
            if (st[leave] == AT_LO && lo[leave] == -INF) st[leave] = (hi[leave] < INF) ? AT_UP : FREE_NB;
            if (st[leave] == AT_UP && hi[leave] == INF) st[leave] = (lo[leave] > -INF) ? AT_LO : FREE_NB;
            x[leave] = nb_value(leave);
            basis[r] = q;
            st[q] = BASIC;
            degenerate = (tstep <= ptol) ? degenerate + 1 : 0;
            if (!push_eta(r, alpha, alnz)) {
                if (!factorize()) return done(S_NUM);
                compute_xb();
                if (!phase1) compute_d();
            }
        }
    }

    // ---------------------------------------------------------------- drivers
    void perturb_costs() {
        double bigc = 0;
        for (int j = 0; j < n; ++j) bigc = std::max(bigc, std::fabs(cost[j]));
        if (bigc > 100) bigc = std::sqrt(std::sqrt(bigc));
        bigc = std::max(bigc, 1.0);
        const double base = 5e-7 * bigc;
        for (int j = 0; j < n; ++j) {
            if (lo[j] == hi[j]) continue;
            if (lo[j] == -INF && hi[j] == INF) continue;
            double xp = (1.0 + std::fabs(cost[j])) * base * (1.0 + rand01());
            int sj = st[j];
            if (sj == BASIC) sj = hi[j] == INF ? AT_LO : (lo[j] == -INF ? AT_UP : (cost[j] >= 0 ? AT_LO : AT_UP));
            cost[j] += (sj == AT_LO) ? xp : -xp;
        }
    }

    // dual phase 1 on the auxiliary box problem; leaves statuses for the model's own bounds
    int dual_phase1() {
        std::vector<double> slo = lo, shi = hi;
        for (int j = 0; j < N; ++j) {
            bool fl = slo[j] > -INF, fu = shi[j] < INF;
            if (fl && fu) { lo[j] = 0; hi[j] = 0; }
            else if (fl) { lo[j] = 0; hi[j] = 1; }
            else if (fu) { lo[j] = -1; hi[j] = 0; }
            else { lo[j] = -1000; hi[j] = 1000; }
        }
        for (int j = 0; j < N; ++j) if (st[j] != BASIC) st[j] = (lo[j] == hi[j] || d[j] >= 0) ? AT_LO : AT_UP;
        compute_xb();
        int res = dual();
        lo = slo; hi = shi;
        for (int j = 0; j < N; ++j) {
            if (st[j] == BASIC) continue;
            bool fl = lo[j] > -INF, fu = hi[j] < INF;
            if (fl && fu) st[j] = (lo[j] == hi[j] || d[j] >= 0) ? AT_LO : AT_UP;
            else if (fl) st[j] = AT_LO;
            else if (fu) st[j] = AT_UP;
            else st[j] = FREE_NB;
        }
        compute_xb();
        return res;
    }

    int solve(bool cold) {
        compute_xb();
        compute_d();
        const std::vector<double> true_cost = cost;
        int res = -1;
        bool pf = primal_feasible(ptol);
        if (pf && count_dual_infeas() == 0) res = S_OPTIMAL;
        // A primal feasible start goes to the primal simplex only when warm. Cold, the slack basis of a
        // model where doing nothing is feasible (a refinery buying no crude) is a maximally degenerate
        // vertex, and the primal simplex has only Bland's rule there: a public-data refinery twin ran
        // past 1.8M pivots cold (3.5k from a warm basis). The perturbed dual simplex below handles it.
        else if (pf && !cold) res = primal(false);
        else {
            for (int j = 0; j < N; ++j)                                   // boxed: take the feasible bound
                if (boxed(j) && dual_infeas(j) > dtol) st[j] = (st[j] == AT_LO) ? AT_UP : AT_LO;
            compute_xb();
            bool dfeas = count_dual_infeas() == 0;
            if (!dfeas) {
                int p1 = dual_phase1();
                cost = true_cost;
                if (p1 == S_TIME || p1 == S_ITER) return p1;
                compute_d();
                dfeas = (p1 == S_OPTIMAL) && count_dual_infeas() == 0;
            }
            if (dfeas) {
                if (cold) { perturb_costs(); compute_d(); correct_duals(); }
                res = dual();
                cost = true_cost;
                if (res == S_TIME || res == S_ITER) return res;
                if (res == S_OPTIMAL || res == S_INFEASIBLE) {
                    if (!factorize()) res = -1;
                    else {
                        recompute();
                        if (res == S_OPTIMAL && !primal_feasible(ptol)) res = -1;
                        if (res == S_OPTIMAL && count_dual_infeas() > 0) res = primal(false);
                    }
                } else res = -1;
            }
        }
        if (res < 0 || res == S_NUM) {                                   // fallback: primal phase 1 + 2
            cost = true_cost;
            if (!factorize()) {
                basis.clear();
                for (int j = 0; j < n; ++j) st[j] = lo[j] > -INF ? AT_LO : (hi[j] < INF ? AT_UP : FREE_NB);
                for (int i = 0; i < m; ++i) { st[n + i] = BASIC; basis.push_back(n + i); }
                factorize();
            }
            compute_xb();
            int p1 = primal(true);
            if (p1 == S_TIME || p1 == S_ITER || p1 == S_NUM) return p1;
            double infeas = 0;
            for (int k = 0; k < m; ++k) {
                int j = basis[k];
                infeas += std::max(0.0, lo[j] - x[j]) + std::max(0.0, x[j] - hi[j]);
            }
            if (infeas > 1e-6 * (1.0 + m)) return S_INFEASIBLE;
            res = primal(false);
        }
        cost = true_cost;
        return res;
    }
};

}  // namespace

extern "C" {

// Profile of the solves so far: seconds and calls for factor, ftran, btran, price, chuzc, chuzr,
// update, other; reset != 0 clears them.
void nr_profile(double* secs, long* calls, int reset) {
    for (int k = 0; k < P_N; ++k) { secs[k] = g_prof[k]; calls[k] = g_count[k]; }
    if (reset) { std::memset(g_prof, 0, sizeof g_prof); std::memset(g_count, 0, sizeof g_count); }
}

// Test hook: factorise the basis given by `basis` (m column indices of [A -I]), then solve
// B z = b and B' pi = d. Returns 0, or 1 if the factorisation reported a singular basis.
int nr_lu_test(int m, int n, const int* Ap, const int* Ai, const double* Ax, const int* basis,
               const double* b, const double* d, double* z_out, double* pi_out) {
    Simplex S;
    S.Ap = Ap; S.Ai = Ai; S.Ax = Ax;
    S.m = m; S.n = n; S.N = n + m;
    S.ybuf.assign(m, 0.0);
    S.basis.assign(basis, basis + m);
    if (!S.factorize()) return 1;
    std::vector<double> z(b, b + m), pi(d, d + m);
    S.ftran(z);
    S.btran(pi);
    for (int i = 0; i < m; ++i) { z_out[i] = z[i]; pi_out[i] = pi[i]; }
    return 0;
}

// status (n+m): in = warm-start statuses (BASIC=0, AT_LO=1, AT_UP=2, FREE_NB=3) or all -1 for the
// slack basis; out = final statuses. Returns S_* and fills x (n), y (m), iterations.
int nr_simplex(int m, int n, const int* Ap, const int* Ai, const double* Ax, const double* c,
               const double* lx, const double* ux, const double* lc, const double* uc, double tol,
               double time_limit, long max_iter, int* status, double* x_out, double* y_out,
               long* iters_out, int* refactors_out) {
    Simplex S;
    S.Ap = Ap; S.Ai = Ai; S.Ax = Ax;
    S.m = m; S.n = n; S.N = n + m;
    S.ptol = std::max(tol, 1e-9); S.dtol = tol;
    S.time_limit = time_limit; S.max_iter = max_iter;
    S.t0 = Clock::now();
    S.ybuf.assign(m, 0.0);
    S.build_rows();
    S.cost.assign(S.N, 0.0);
    S.lo.resize(S.N); S.hi.resize(S.N); S.x.assign(S.N, 0.0); S.d.assign(S.N, 0.0); S.st.assign(S.N, AT_LO);
    S.dse.assign(m, 1.0);
    for (int j = 0; j < n; ++j) { S.cost[j] = c[j]; S.lo[j] = lx[j]; S.hi[j] = ux[j]; }
    for (int i = 0; i < m; ++i) { S.lo[n + i] = lc[i]; S.hi[n + i] = uc[i]; }
    auto pick = [&](int j) { return S.lo[j] > -INF ? AT_LO : (S.hi[j] < INF ? AT_UP : FREE_NB); };
    bool warm = status[0] >= 0;
    S.basis.clear();
    if (warm) {
        for (int j = 0; j < S.N; ++j) {
            S.st[j] = status[j];
            if (S.st[j] == BASIC) S.basis.push_back(j);
            else if (S.st[j] == AT_LO && S.lo[j] == -INF) S.st[j] = pick(j);
            else if (S.st[j] == AT_UP && S.hi[j] == INF) S.st[j] = pick(j);
        }
        if ((int)S.basis.size() != m) warm = false;
    }
    if (warm && !S.factorize()) warm = false;
    if (!warm) {
        S.basis.clear();
        for (int j = 0; j < n; ++j) S.st[j] = pick(j);
        for (int i = 0; i < m; ++i) { S.st[n + i] = BASIC; S.basis.push_back(n + i); }
        S.factorize();
    }
    int res = S.solve(!warm);
    std::vector<double> pi;
    if (S.factorize()) S.compute_xb();
    S.compute_pi(S.cost, pi);
    for (int j = 0; j < n; ++j) x_out[j] = S.x[j];
    for (int i = 0; i < m; ++i) y_out[i] = pi[i];
    for (int j = 0; j < S.N; ++j) status[j] = S.st[j];
    *iters_out = S.iters;
    *refactors_out = S.refactors;
    return res;
}

}  // extern "C"
