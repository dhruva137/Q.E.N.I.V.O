// Nested dissection by multilevel graph partition.
//
// A. George, "Nested dissection of a regular finite element mesh", SIAM Journal on
// Numerical Analysis 10 (1973) 345-363: order each part, then the vertex separator.
// Coarsening is heavy-edge matching (G. Karypis and V. Kumar, SIAM J. Sci. Comput. 20
// (1998) 359-392). The coarsest graph is improved by pair swaps (B. W. Kernighan and
// S. Lin, Bell System Technical Journal 49 (1970) 291-307) and every level is refined
// by gain buckets (C. M. Fiduccia and R. M. Mattheyses, 19th Design Automation
// Conference, 1982). The separator is the boundary of one side of the edge cut.
// Small or poorly separated blocks fall back to approximate minimum degree.

#include "ipm_internal.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <list>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qenivo {
namespace {

struct WGraph {
    int n = 0;
    std::vector<int> vertex_weight;
    std::vector<std::vector<std::pair<int, int>>> adj;
};

int edge_between(const WGraph& g, int a, int b) {
    const auto& row = g.adj[a];
    if (row.size() > g.adj[b].size()) {
        return edge_between(g, b, a);
    }
    for (const auto& e : row) {
        if (e.first == b) {
            return e.second;
        }
    }
    return 0;
}

int cut_size(const WGraph& g, const std::vector<int>& side) {
    int cut = 0;
    for (int v = 0; v < g.n; ++v) {
        for (const auto& e : g.adj[v]) {
            if (v < e.first && side[v] != side[e.first]) {
                cut += e.second;
            }
        }
    }
    return cut;
}

std::vector<int> gains_of(const WGraph& g, const std::vector<int>& side) {
    std::vector<int> gain(g.n, 0);
    for (int v = 0; v < g.n; ++v) {
        int gsum = 0;
        for (const auto& e : g.adj[v]) {
            if (side[e.first] != side[v]) {
                gsum += e.second;
            } else {
                gsum -= e.second;
            }
        }
        gain[v] = gsum;
    }
    return gain;
}

void kernighan_lin(const WGraph& g, std::vector<int>& side) {
    if (g.n < 2) {
        return;
    }
    for (int pass = 0; pass < 4; ++pass) {
        std::vector<char> locked(g.n, 0);
        std::vector<std::pair<int, int>> swaps;
        int delta = 0;
        int best_delta = 0;
        int best_len = 0;
        const int steps = std::max(1, g.n / 2);
        for (int step = 0; step < steps; ++step) {
            std::vector<int> gain = gains_of(g, side);
            int best = INT_MIN;
            int ba = -1;
            int bb = -1;
            for (int a = 0; a < g.n; ++a) {
                if (locked[a] || side[a] != 0) {
                    continue;
                }
                for (int b = 0; b < g.n; ++b) {
                    if (locked[b] || side[b] != 1) {
                        continue;
                    }
                    int gpair = gain[a] + gain[b] - 2 * edge_between(g, a, b);
                    if (gpair > best || (gpair == best && (ba < 0 || a < ba || (a == ba && b < bb)))) {
                        best = gpair;
                        ba = a;
                        bb = b;
                    }
                }
            }
            if (ba < 0) {
                break;
            }
            side[ba] = 1;
            side[bb] = 0;
            locked[ba] = 1;
            locked[bb] = 1;
            delta += best;
            swaps.push_back({ba, bb});
            if (delta > best_delta) {
                best_delta = delta;
                best_len = (int)swaps.size();
            }
        }
        for (int s = (int)swaps.size() - 1; s >= best_len; --s) {
            side[swaps[s].first] = 0;
            side[swaps[s].second] = 1;
        }
        if (best_len == 0) {
            break;
        }
    }
}

void fiduccia_mattheyses(const WGraph& g, std::vector<int>& side) {
    if (g.n < 2) {
        return;
    }
    int total = 0;
    for (int w : g.vertex_weight) {
        total += w;
    }
    if (total <= 0) {
        total = g.n;
    }
    const int slack = std::max(1, total / 5);
    int gmax = 1;
    for (int v = 0; v < g.n; ++v) {
        int s = 0;
        for (const auto& e : g.adj[v]) {
            s += e.second;
        }
        if (s > gmax) {
            gmax = s;
        }
    }
    if (gmax > 1000000) {
        gmax = 1000000;
    }
    for (int pass = 0; pass < 8; ++pass) {
        std::vector<int> gain = gains_of(g, side);
        for (int& gv : gain) {
            if (gv > gmax) {
                gv = gmax;
            }
            if (gv < -gmax) {
                gv = -gmax;
            }
        }
        std::vector<char> locked(g.n, 0);
        std::vector<std::list<int>> bucket(2 * gmax + 1);
        std::vector<std::list<int>::iterator> loc(g.n);
        std::vector<int> bucket_of(g.n, 0);
        auto place = [&](int v) {
            int b = gain[v] + gmax;
            bucket[b].push_front(v);
            loc[v] = bucket[b].begin();
            bucket_of[v] = b;
        };
        for (int v = 0; v < g.n; ++v) {
            place(v);
        }
        int w0 = 0;
        for (int v = 0; v < g.n; ++v) {
            if (side[v] == 0) {
                w0 += g.vertex_weight[v];
            }
        }
        const int cut0 = cut_size(g, side);
        int cut = cut0;
        int best_cut = cut0;
        std::vector<int> best = side;
        int cursor = 2 * gmax;
        bool moved = false;
        for (int step = 0; step < g.n; ++step) {
            int pick = -1;
            while (cursor >= 0 && bucket[cursor].empty()) {
                --cursor;
            }
            for (int b = cursor; b >= 0 && pick < 0; --b) {
                for (int v : bucket[b]) {
                    if (locked[v]) {
                        continue;
                    }
                    int nw = w0 + (side[v] == 0 ? -g.vertex_weight[v] : g.vertex_weight[v]);
                    int lo = total / 2 - slack;
                    int hi = total / 2 + slack;
                    if (nw < lo || nw > hi) {
                        continue;
                    }
                    pick = v;
                    break;
                }
            }
            if (pick < 0) {
                break;
            }
            const int old = side[pick];
            const int g0 = gain[pick];
            bucket[bucket_of[pick]].erase(loc[pick]);
            locked[pick] = 1;
            side[pick] = 1 - old;
            w0 += (old == 0 ? -g.vertex_weight[pick] : g.vertex_weight[pick]);
            cut -= g0;
            moved = true;
            for (const auto& e : g.adj[pick]) {
                int u = e.first;
                if (locked[u]) {
                    continue;
                }
                int delta = (side[u] == old ? 2 * e.second : -2 * e.second);
                int ng = gain[u] + delta;
                if (ng > gmax) {
                    ng = gmax;
                }
                if (ng < -gmax) {
                    ng = -gmax;
                }
                bucket[bucket_of[u]].erase(loc[u]);
                gain[u] = ng;
                place(u);
                if (bucket_of[u] > cursor) {
                    cursor = bucket_of[u];
                }
            }
            if (cut < best_cut) {
                best_cut = cut;
                best = side;
            }
        }
        side.swap(best);
        if (!moved || best_cut >= cut0) {
            break;
        }
    }
}

std::vector<int> initial_side(const WGraph& g) {
    const int n = g.n;
    std::vector<int> side(n, 1);
    if (n == 0) {
        return side;
    }
    auto bfs = [&](int src) {
        std::vector<int> dist(n, -1);
        std::vector<int> q;
        q.push_back(src);
        dist[src] = 0;
        for (size_t h = 0; h < q.size(); ++h) {
            int v = q[h];
            for (const auto& e : g.adj[v]) {
                if (dist[e.first] < 0) {
                    dist[e.first] = dist[v] + 1;
                    q.push_back(e.first);
                }
            }
        }
        return q;
    };
    std::vector<int> reach = bfs(0);
    int far = reach.empty() ? 0 : reach.back();
    reach = bfs(far);
    std::vector<char> seen(n, 0);
    std::vector<int> order;
    order.reserve(n);
    for (int v : reach) {
        if (!seen[v]) {
            seen[v] = 1;
            order.push_back(v);
        }
    }
    for (int v = 0; v < n; ++v) {
        if (!seen[v]) {
            for (int u : bfs(v)) {
                if (!seen[u]) {
                    seen[u] = 1;
                    order.push_back(u);
                }
            }
        }
    }
    int total = 0;
    for (int w : g.vertex_weight) {
        total += w;
    }
    int half = std::max(1, total / 2);
    int acc = 0;
    for (int v : order) {
        if (acc < half) {
            side[v] = 0;
            acc += g.vertex_weight[v];
        } else {
            side[v] = 1;
        }
    }
    if (acc == 0 || acc == total) {
        side[order.empty() ? 0 : order[0]] = 0;
        if (n > 1) {
            side[order.size() > 1 ? order[1] : 1] = 1;
        }
    }
    return side;
}

struct Coarse {
    WGraph graph;
    std::vector<int> map;
};

Coarse coarsen(const WGraph& g) {
    const int n = g.n;
    std::vector<int> match(n, -1);
    std::vector<int> visit(n);
    for (int i = 0; i < n; ++i) {
        visit[i] = i;
    }
    std::sort(visit.begin(), visit.end(), [&](int a, int b) {
        int da = 0;
        int db = 0;
        for (const auto& e : g.adj[a]) {
            da += e.second;
        }
        for (const auto& e : g.adj[b]) {
            db += e.second;
        }
        if (da != db) {
            return da > db;
        }
        return a < b;
    });
    for (int v : visit) {
        if (match[v] != -1) {
            continue;
        }
        int best = -1;
        int bw = -1;
        for (const auto& e : g.adj[v]) {
            if (match[e.first] == -1 && (e.second > bw || (e.second == bw && (best < 0 || e.first < best)))) {
                best = e.first;
                bw = e.second;
            }
        }
        if (best >= 0) {
            match[v] = best;
            match[best] = v;
        } else {
            match[v] = v;
        }
    }
    std::vector<int> map(n, -1);
    int nc = 0;
    for (int v = 0; v < n; ++v) {
        if (map[v] >= 0) {
            continue;
        }
        int id = nc++;
        map[v] = id;
        if (match[v] != v) {
            map[match[v]] = id;
        }
    }
    std::vector<std::unordered_map<int, int>> em(nc);
    for (int v = 0; v < n; ++v) {
        for (const auto& e : g.adj[v]) {
            if (v >= e.first) {
                continue;
            }
            int cv = map[v];
            int cu = map[e.first];
            if (cv == cu) {
                continue;
            }
            em[cv][cu] += e.second;
        }
    }
    Coarse out;
    out.map = std::move(map);
    out.graph.n = nc;
    out.graph.vertex_weight.assign(nc, 0);
    for (int v = 0; v < n; ++v) {
        out.graph.vertex_weight[out.map[v]] += g.vertex_weight[v];
    }
    out.graph.adj.assign(nc, {});
    for (int i = 0; i < nc; ++i) {
        for (const auto& kv : em[i]) {
            out.graph.adj[i].push_back({kv.first, kv.second});
            out.graph.adj[kv.first].push_back({i, kv.second});
        }
    }
    return out;
}

std::vector<int> bisect(WGraph graph) {
    std::vector<WGraph> levels;
    std::vector<std::vector<int>> maps;
    levels.push_back(std::move(graph));
    while (levels.back().n > 36) {
        Coarse c = coarsen(levels.back());
        if (c.graph.n < 2 || c.graph.n >= (int)(levels.back().n * 9 / 10)) {
            break;
        }
        maps.push_back(std::move(c.map));
        levels.push_back(std::move(c.graph));
    }
    std::vector<int> side = initial_side(levels.back());
    if (levels.back().n <= 48) {
        kernighan_lin(levels.back(), side);
    }
    fiduccia_mattheyses(levels.back(), side);
    for (int level = (int)levels.size() - 2; level >= 0; --level) {
        std::vector<int> up(levels[level].n, 0);
        const std::vector<int>& map = maps[level];
        for (int i = 0; i < levels[level].n; ++i) {
            up[i] = side[map[i]];
        }
        fiduccia_mattheyses(levels[level], up);
        side.swap(up);
    }
    return side;
}

WGraph induced(const std::vector<std::vector<int>>& adj, const std::vector<int>& nodes) {
    const int n = (int)nodes.size();
    std::vector<int> local(adj.size(), -1);
    for (int i = 0; i < n; ++i) {
        local[nodes[i]] = i;
    }
    WGraph g;
    g.n = n;
    g.vertex_weight.assign(n, 1);
    g.adj.assign(n, {});
    for (int i = 0; i < n; ++i) {
        for (int nb : adj[nodes[i]]) {
            int j = (nb >= 0 && nb < (int)local.size()) ? local[nb] : -1;
            if (j > i) {
                g.adj[i].push_back({j, 1});
                g.adj[j].push_back({i, 1});
            }
        }
    }
    return g;
}

void append_amd(const std::vector<std::vector<int>>& adj, const std::vector<int>& nodes, std::vector<int>& order) {
    if (nodes.empty()) {
        return;
    }
    if (nodes.size() == 1) {
        order.push_back(nodes[0]);
        return;
    }
    std::vector<int> local_of(adj.size(), -1);
    std::vector<std::vector<int>> sub(nodes.size());
    for (int i = 0; i < (int)nodes.size(); ++i) {
        local_of[nodes[i]] = i;
    }
    for (int i = 0; i < (int)nodes.size(); ++i) {
        for (int nb : adj[nodes[i]]) {
            int j = (nb >= 0 && nb < (int)local_of.size()) ? local_of[nb] : -1;
            if (j >= 0 && j != i) {
                sub[i].push_back(j);
            }
        }
        std::sort(sub[i].begin(), sub[i].end());
        sub[i].erase(std::unique(sub[i].begin(), sub[i].end()), sub[i].end());
    }
    std::vector<int> local_order = amd_from_adj(sub);
    for (int id : local_order) {
        order.push_back(nodes[id]);
    }
}

void dissect_block(const std::vector<std::vector<int>>& adj, const std::vector<int>& nodes, std::vector<int>& order) {
    if ((int)nodes.size() <= 12) {
        append_amd(adj, nodes, order);
        return;
    }
    WGraph g = induced(adj, nodes);
    std::vector<int> side = bisect(g);
    std::vector<char> boundary(g.n, 0);
    int bcount = 0;
    for (int v = 0; v < g.n; ++v) {
        if (side[v] != 0) {
            continue;
        }
        for (const auto& e : g.adj[v]) {
            if (side[e.first] == 1) {
                boundary[v] = 1;
                ++bcount;
                break;
            }
        }
    }
    int other = 0;
    std::vector<char> boundary_b(g.n, 0);
    for (int v = 0; v < g.n; ++v) {
        if (side[v] != 1) {
            continue;
        }
        for (const auto& e : g.adj[v]) {
            if (side[e.first] == 0) {
                boundary_b[v] = 1;
                ++other;
                break;
            }
        }
    }
    if (other > 0 && other < bcount) {
        boundary.swap(boundary_b);
        bcount = other;
    }
    std::vector<int> part_a;
    std::vector<int> part_b;
    std::vector<int> sep;
    for (int i = 0; i < g.n; ++i) {
        if (boundary[i]) {
            sep.push_back(nodes[i]);
        } else if (side[i] == 0) {
            part_a.push_back(nodes[i]);
        } else {
            part_b.push_back(nodes[i]);
        }
    }
    if (part_a.empty() || part_b.empty() || bcount * 2 > g.n) {
        append_amd(adj, nodes, order);
        return;
    }
    dissect_block(adj, part_a, order);
    dissect_block(adj, part_b, order);
    append_amd(adj, sep, order);
}

}  // namespace

std::vector<int> nd_from_adj(const std::vector<std::vector<int>>& adj) {
    const int n = (int)adj.size();
    std::vector<int> nodes(n);
    for (int i = 0; i < n; ++i) {
        nodes[i] = i;
    }
    std::vector<int> order;
    order.reserve(n);
    dissect_block(adj, nodes, order);
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
