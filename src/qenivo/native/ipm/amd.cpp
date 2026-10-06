// Approximate minimum degree on the quotient graph.
//
// P. R. Amestoy, T. A. Davis and I. S. Duff, "An Approximate Minimum Degree Ordering
// Algorithm", SIAM Journal on Matrix Analysis and Applications 17 (1996) 886-905.
//
// The quotient graph stores variable–variable edges that are not covered by elements.
// When a pivot is eliminated its neighbours become an element (a clique that is not
// materialised). The external degree of a neighbour is the Amestoy–Davis–Duff upper
// bound: the weight of the uncovered neighbours plus, for each adjacent element, the
// element weight minus the pivot weight. Overlapping elements are not expanded, which
// is the approximation. An independent set of currently minimum-degree variables is
// eliminated before the next degree sweep (multiple elimination). Identical external
// adjacency lists are merged into one supervariable.

#include "ipm_internal.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qenivo {

bool is_permutation(const std::vector<int>& perm, int n) {
    if ((int)perm.size() != n) {
        return false;
    }
    std::vector<char> seen(n, 0);
    for (int v : perm) {
        if (v < 0 || v >= n || seen[v]) {
            return false;
        }
        seen[v] = 1;
    }
    return true;
}

std::vector<std::vector<int>> adjacency(int n, const int* colptr, const int* rowidx) {
    std::vector<std::vector<int>> adj(std::max(n, 0));
    if (n <= 0 || colptr == nullptr || rowidx == nullptr) {
        return adj;
    }
    for (int j = 0; j < n; ++j) {
        for (int p = colptr[j]; p < colptr[j + 1]; ++p) {
            int i = rowidx[p];
            if (i == j || i < 0 || i >= n) {
                continue;
            }
            adj[j].push_back(i);
            adj[i].push_back(j);
        }
    }
    for (int i = 0; i < n; ++i) {
        auto& a = adj[i];
        std::sort(a.begin(), a.end());
        a.erase(std::unique(a.begin(), a.end()), a.end());
    }
    return adj;
}

namespace {

void sort_unique(std::vector<int>& v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

uint64_t hash_lists(const std::vector<int>& a, const std::vector<int>& e) {
    uint64_t h = 1469598103934665603ull;
    for (int v : a) {
        h ^= (uint64_t)(v + 1) + 0x9e3779b97f4a7c15ull;
        h *= 1099511628211ull;
    }
    h ^= 0x9e3779b97f4a7c15ull;
    for (int v : e) {
        h ^= (uint64_t)(v + 1) + 0xbf58476d1ce4e5b9ull;
        h *= 1099511628211ull;
    }
    return h;
}

}  // namespace

std::vector<int> amd_from_adj(const std::vector<std::vector<int>>& adj) {
    const int n = (int)adj.size();
    std::vector<int> order;
    order.reserve(n);
    if (n == 0) {
        return order;
    }
    std::vector<std::vector<int>> A = adj;
    std::vector<std::vector<int>> E(n);
    std::vector<std::vector<int>> elem_vars(n);
    std::vector<int> elem_w(n, 0);
    std::vector<int> weight(n, 1);
    std::vector<int> rep(n);
    std::iota(rep.begin(), rep.end(), 0);
    std::vector<char> alive(n, 1);
    std::vector<std::vector<int>> members(n);
    for (int i = 0; i < n; ++i) {
        members[i].push_back(i);
    }
    std::vector<int> degree(n, 0);
    for (int i = 0; i < n; ++i) {
        degree[i] = (int)A[i].size();
    }
    std::vector<int> mark(n, 0);
    int stamp = 1;
    int remain = n;

    auto find = [&](int v) {
        int r = v;
        while (rep[r] != r) {
            r = rep[r];
        }
        while (v != r) {
            int nxt = rep[v];
            rep[v] = r;
            v = nxt;
        }
        return r;
    };

    auto approx = [&](int i) {
        long long d = 0;
        for (int v : A[i]) {
            d += weight[v];
        }
        for (int e : E[i]) {
            d += (long long)elem_w[e] - weight[i];
        }
        if (d < 0) {
            d = 0;
        }
        if (d > remain - weight[i]) {
            d = remain - weight[i];
        }
        if (d > n) {
            d = n;
        }
        return (int)d;
    };

    auto eliminate = [&](int p) {
        if (stamp == 0x7fffffff) {
            std::fill(mark.begin(), mark.end(), 0);
            stamp = 1;
        }
        ++stamp;
        std::vector<int> clique;
        auto add = [&](int v) {
            v = find(v);
            if (!alive[v] || v == p || mark[v] == stamp) {
                return;
            }
            mark[v] = stamp;
            clique.push_back(v);
        };
        for (int v : A[p]) {
            add(v);
        }
        for (int e : E[p]) {
            for (int v : elem_vars[e]) {
                add(v);
            }
        }
        std::vector<int> absorbed = E[p];
        std::vector<char> absorbed_mark(n, 0);
        for (int e : absorbed) {
            if (e >= 0 && e < n) {
                absorbed_mark[e] = 1;
            }
        }
        elem_vars[p] = clique;
        int ew = 0;
        for (int v : clique) {
            ew += weight[v];
        }
        elem_w[p] = ew;
        for (int i : clique) {
            std::vector<int> na;
            na.reserve(A[i].size());
            for (int v : A[i]) {
                v = find(v);
                if (!alive[v] || v == i || v == p || mark[v] == stamp) {
                    continue;
                }
                na.push_back(v);
            }
            sort_unique(na);
            A[i].swap(na);
            std::vector<int> ne;
            ne.reserve(E[i].size() + 1);
            for (int e : E[i]) {
                if (e == p || (e >= 0 && e < n && absorbed_mark[e])) {
                    continue;
                }
                ne.push_back(e);
            }
            ne.push_back(p);
            sort_unique(ne);
            E[i].swap(ne);
            degree[i] = approx(i);
        }
        std::unordered_map<uint64_t, std::vector<int>> buckets;
        for (int i : clique) {
            if (!alive[i]) {
                continue;
            }
            buckets[hash_lists(A[i], E[i])].push_back(i);
        }
        for (auto& bucket : buckets) {
            auto& ids = bucket.second;
            for (size_t a = 0; a < ids.size(); ++a) {
                int i = ids[a];
                if (!alive[i]) {
                    continue;
                }
                for (size_t b = a + 1; b < ids.size(); ++b) {
                    int j = ids[b];
                    if (!alive[j]) {
                        continue;
                    }
                    if (A[i] == A[j] && E[i] == E[j]) {
                        weight[i] += weight[j];
                        members[i].insert(members[i].end(), members[j].begin(), members[j].end());
                        members[j].clear();
                        alive[j] = 0;
                        rep[j] = i;
                        degree[i] = approx(i);
                    }
                }
            }
        }
        alive[p] = 0;
        std::sort(members[p].begin(), members[p].end());
        for (int v : members[p]) {
            order.push_back(v);
        }
        remain -= (int)members[p].size();
        if (remain < 0) {
            remain = 0;
        }
    };

    int guard = 0;
    while (guard++ < n + 2) {
        int dmin = n + 1;
        int any = -1;
        for (int i = 0; i < n; ++i) {
            if (!alive[i]) {
                continue;
            }
            any = i;
            if (degree[i] < dmin) {
                dmin = degree[i];
            }
        }
        if (any < 0) {
            break;
        }
        std::vector<char> blocked(n, 0);
        std::vector<int> batch;
        for (int i = 0; i < n; ++i) {
            if (!alive[i] || degree[i] != dmin || blocked[i]) {
                continue;
            }
            batch.push_back(i);
            blocked[i] = 1;
            for (int v : A[i]) {
                v = find(v);
                if (v >= 0 && v < n) {
                    blocked[v] = 1;
                }
            }
            for (int e : E[i]) {
                for (int v : elem_vars[e]) {
                    v = find(v);
                    if (v >= 0 && v < n) {
                        blocked[v] = 1;
                    }
                }
            }
            if ((int)batch.size() >= 32) {
                break;
            }
        }
        if (batch.empty()) {
            break;
        }
        for (int p : batch) {
            p = find(p);
            if (p < 0 || p >= n || !alive[p]) {
                continue;
            }
            eliminate(p);
        }
    }
    std::vector<char> seen(n, 0);
    std::vector<int> cleaned;
    cleaned.reserve(n);
    for (int v : order) {
        if (v >= 0 && v < n && !seen[v]) {
            seen[v] = 1;
            cleaned.push_back(v);
        }
    }
    for (int i = 0; i < n; ++i) {
        if (!seen[i]) {
            cleaned.push_back(i);
        }
    }
    return cleaned;
}

}  // namespace qenivo
