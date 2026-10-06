/* Native MPS reader: free and fixed format, RANGES, BOUNDS, MARKER, OBJSENSE, QUADOBJ/QMATRIX.
 * Compiled on first use (see io/native_mps.py). Gzip is decompressed in Python before this runs.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  define NR_EXPORT extern "C" __declspec(dllexport)
#else
#  define NR_EXPORT extern "C"
#endif

namespace {

constexpr int kObj = -1;
constexpr int kDrop = -2;
constexpr double kInf = std::numeric_limits<double>::infinity();

struct Triplet { int i, j; double v; };
struct BoundOp { std::string typ; int j; double v; bool has_v; };
struct QEntry { int i, j; double v; bool mirror; };

struct ParseState {
    std::unordered_map<std::string, int> row_idx, col_idx;
    std::vector<std::string> row_names, col_names, row_types;
    std::vector<double> cost;
    std::vector<char> is_int;
    std::vector<Triplet> trips;
    std::unordered_map<int, double> rhs, rng;
    std::vector<BoundOp> bnd_ops;
    std::vector<QEntry> q_entries;
    std::vector<std::string> notes;
    double c0 = 0.0, sense = 1.0;
    std::string rhs_set, rng_set, bnd_set, objname_hint, prob_name;
    int obj_row = -999;
    bool in_int = false;
    bool name_locked = false;
    std::string section;
    std::string errmsg;
};

static inline void trim_inplace(std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r')) ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) --b;
    s = s.substr(a, b - a);
}

static std::vector<std::string> split_ws(const std::string& line) {
    std::vector<std::string> out;
    size_t i = 0, n = line.size();
    while (i < n) {
        while (i < n && (line[i] == ' ' || line[i] == '\t')) ++i;
        if (i >= n) break;
        size_t j = i;
        while (j < n && line[j] != ' ' && line[j] != '\t') ++j;
        out.emplace_back(line.substr(i, j - i));
        i = j;
    }
    return out;
}

static std::vector<std::string> fixed_fields(const std::string& line) {
    // Columns 2-3, 5-12, 15-22, 25-36, 40-47, 50-61 (1-based inclusive) -> 0-based half-open.
    static const int spans[][2] = {{1, 3}, {4, 12}, {14, 22}, {24, 36}, {39, 47}, {49, 61}};
    std::vector<std::string> out;
    std::string padded = line;
    if (padded.size() < 61) padded.resize(61, ' ');
    for (auto& sp : spans) {
        std::string f = padded.substr(sp[0], sp[1] - sp[0]);
        trim_inplace(f);
        if (!f.empty()) out.push_back(f);
    }
    return out;
}

static std::string upper(std::string s) {
    for (char& c : s) if (c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
    return s;
}

static bool parse_float(const std::string& s, double& v) {
    try {
        size_t idx = 0;
        v = std::stod(s, &idx);
        return idx > 0;
    } catch (...) {
        return false;
    }
}

static void fail(ParseState& st, const std::string& msg) { st.errmsg = msg; }

static int ensure_col(ParseState& st, const std::string& cname) {
    auto it = st.col_idx.find(cname);
    if (it != st.col_idx.end()) return it->second;
    int j = (int)st.col_names.size();
    st.col_idx[cname] = j;
    st.col_names.push_back(cname);
    st.cost.push_back(0.0);
    st.is_int.push_back(st.in_int ? 1 : 0);
    return j;
}

static bool handle_columns(ParseState& st, const std::string& line, std::vector<std::string> t) {
    size_t nt = t.size();
    if (nt >= 3 && t[1] == "'MARKER'") {
        for (size_t k = 2; k < nt; ++k) {
            if (t[k].find("INTORG") != std::string::npos) st.in_int = true;
            else if (t[k].find("INTEND") != std::string::npos) st.in_int = false;
        }
        return true;
    }
    auto row_ok = [&](const std::vector<std::string>& tok) -> bool {
        if (tok.size() < 3 || (tok.size() % 2) == 0) return false;
        for (size_t k = 1; k + 1 < tok.size(); k += 2)
            if (!st.row_idx.count(tok[k])) return false;
        return true;
    };
    if (!row_ok(t)) {
        auto tf = fixed_fields(line);
        if (row_ok(tf)) t = std::move(tf);
    }
    if (t.size() < 3) return true;
    int j = ensure_col(st, t[0]);
    for (size_t k = 1; k + 1 < t.size(); k += 2) {
        auto rit = st.row_idx.find(t[k]);
        if (rit == st.row_idx.end()) {
            auto tf = fixed_fields(line);
            if (k < tf.size() && st.row_idx.count(tf[k])) {
                t = std::move(tf);
                rit = st.row_idx.find(t[k]);
            }
        }
        if (rit == st.row_idx.end()) continue;
        double v;
        if (k + 1 >= t.size() || !parse_float(t[k + 1], v)) continue;
        int i = rit->second;
        if (i >= 0) st.trips.push_back({i, j, v});
        else if (i == kObj) st.cost[j] += v;
    }
    return true;
}

static bool handle_rows(ParseState& st, const std::string& line, std::vector<std::string> t) {
    if (t.size() != 2) t = fixed_fields(line);
    if (t.size() < 2) return true;
    std::string typ = upper(t[0]), rname = t[1];
    if (typ == "N") {
        if (st.obj_row == -999 && (st.objname_hint.empty() || rname == st.objname_hint)) {
            st.obj_row = 0;
            st.row_idx[rname] = kObj;
        } else {
            st.row_idx[rname] = kDrop;
        }
    } else {
        st.row_idx[rname] = (int)st.row_names.size();
        st.row_names.push_back(rname);
        st.row_types.push_back(typ);
    }
    return true;
}

static bool handle_rhs_ranges(ParseState& st, const std::string& line, std::vector<std::string> t, bool is_rhs) {
    if (t.size() < 2) t = fixed_fields(line);
    else {
        std::vector<std::string> body = (t.size() % 2 == 1) ? std::vector<std::string>(t.begin() + 1, t.end()) : t;
        bool bad = false;
        for (size_t k = 0; k + 1 < body.size(); k += 2)
            if (!st.row_idx.count(body[k])) { bad = true; break; }
        if (bad) t = fixed_fields(line);
    }
    std::vector<std::string> pairs;
    if (t.size() % 2 == 1) {
        std::string setname = t[0];
        if (is_rhs) {
            if (st.rhs_set.empty()) st.rhs_set = setname;
            if (setname != st.rhs_set) return true;
        } else {
            if (st.rng_set.empty()) st.rng_set = setname;
            if (setname != st.rng_set) return true;
        }
        pairs.assign(t.begin() + 1, t.end());
    } else {
        pairs = t;
    }
    for (size_t k = 0; k + 1 < pairs.size(); k += 2) {
        auto it = st.row_idx.find(pairs[k]);
        if (it == st.row_idx.end()) {
            fail(st, std::string(is_rhs ? "RHS" : "RANGES") + " references unknown row " + pairs[k]);
            return false;
        }
        double v;
        if (!parse_float(pairs[k + 1], v)) continue;
        int i = it->second;
        if (is_rhs) {
            if (i >= 0) st.rhs[i] = v;
            else if (i == kObj) st.c0 = -v;
        } else if (i >= 0) {
            st.rng[i] = v;
        }
    }
    return true;
}

static bool handle_bounds(ParseState& st, const std::string& line, std::vector<std::string> t) {
    if (t.empty()) return true;
    std::string typ = upper(t[0]);
    std::string setname, cname;
    double val = 0.0;
    bool has_v = false;
    if (t.size() >= 3 && st.col_idx.count(t[2])) {
        setname = t[1]; cname = t[2];
        if (t.size() > 3) { parse_float(t[3], val); has_v = true; }
    } else if (t.size() >= 2 && st.col_idx.count(t[1])) {
        cname = t[1];
        if (t.size() > 2) { parse_float(t[2], val); has_v = true; }
    } else {
        t = fixed_fields(line);
        if (t.size() < 3) return true;
        setname = t[1]; cname = t[2];
        if (t.size() > 3) { parse_float(t[3], val); has_v = true; }
    }
    if (!setname.empty()) {
        if (st.bnd_set.empty()) st.bnd_set = setname;
        if (setname != st.bnd_set) return true;
    }
    auto it = st.col_idx.find(cname);
    if (it == st.col_idx.end()) {
        fail(st, "BOUNDS references unknown column " + cname);
        return false;
    }
    if (typ == "SC") {
        fail(st, "bound type SC not supported");
        return false;
    }
    st.bnd_ops.push_back({typ, it->second, has_v ? val : 0.0, has_v});
    return true;
}

static bool handle_quad(ParseState& st, std::vector<std::string>& t, bool mirror) {
    if (t.size() < 3) return true;
    auto iit = st.col_idx.find(t[0]), jit = st.col_idx.find(t[1]);
    if (iit == st.col_idx.end() || jit == st.col_idx.end()) {
        fail(st, "QUAD section references unknown column");
        return false;
    }
    double v;
    if (!parse_float(t[2], v)) return true;
    st.q_entries.push_back({iit->second, jit->second, v, mirror});
    return true;
}

static bool parse_mps(ParseState& st, const char* text, int text_len) {
    const char* p = text;
    const char* end = text + text_len;
    while (p < end) {
        const char* line_start = p;
        while (p < end && *p != '\n') ++p;
        std::string raw(line_start, p - line_start);
        if (p < end && *p == '\n') ++p;
        if (!raw.empty() && raw.back() == '\r') raw.pop_back();
        if (raw.empty() || raw[0] == '*') continue;
        if (raw[0] != ' ' && raw[0] != '\t') {
            auto tok = split_ws(raw);
            if (tok.empty()) continue;
            std::string head = upper(tok[0]);
            st.section = head;
            if (head == "NAME" && tok.size() > 1 && !st.name_locked) st.prob_name = tok[1];
            else if (head == "OBJSENSE" && tok.size() > 1) {
                st.sense = (upper(tok[1]).rfind("MAX", 0) == 0) ? -1.0 : 1.0;
                st.section.clear();
            } else if (head == "OBJSENSE") {
                st.section = "OBJSENSE";
            } else if (head == "OBJNAME" && tok.size() > 1) {
                st.objname_hint = tok[1];
            } else if (head == "SOS" || head == "QCMATRIX" || head == "INDICATORS" || head == "GENERAL") {
                fail(st, "MPS section " + head + " not supported");
                return false;
            } else if (head == "ENDATA") {
                break;
            }
            continue;
        }
        auto t = split_ws(raw);
        if (st.section == "COLUMNS") {
            if (!handle_columns(st, raw, std::move(t))) return false;
        } else if (st.section == "ROWS") {
            if (!handle_rows(st, raw, std::move(t))) return false;
        } else if (st.section == "RHS") {
            if (!handle_rhs_ranges(st, raw, std::move(t), true)) return false;
        } else if (st.section == "RANGES") {
            if (!handle_rhs_ranges(st, raw, std::move(t), false)) return false;
        } else if (st.section == "BOUNDS") {
            if (!handle_bounds(st, raw, std::move(t))) return false;
        } else if (st.section == "QUADOBJ" || st.section == "QSECTION") {
            if (!handle_quad(st, t, true)) return false;
        } else if (st.section == "QMATRIX") {
            if (!handle_quad(st, t, false)) return false;
        } else if (st.section == "OBJSENSE") {
            if (!t.empty()) st.sense = (upper(t[0]).rfind("MAX", 0) == 0) ? -1.0 : 1.0;
        }
    }
    return true;
}

static void build_csr(const std::vector<Triplet>& trips, int m, int n,
                      std::vector<int>& indptr, std::vector<int>& indices, std::vector<double>& data) {
    // Aggregate duplicates then CSR.
    std::unordered_map<long long, double> acc;
    acc.reserve(trips.size() * 2 + 1);
    for (auto& t : trips) {
        long long key = (long long)t.i * (long long)n + t.j;
        acc[key] += t.v;
    }
    std::vector<std::vector<std::pair<int, double>>> rows(m);
    for (auto& kv : acc) {
        if (kv.second == 0.0) continue;
        int i = (int)(kv.first / n), j = (int)(kv.first % n);
        rows[i].push_back({j, kv.second});
    }
    indptr.assign(m + 1, 0);
    for (int i = 0; i < m; ++i) {
        auto& r = rows[i];
        std::sort(r.begin(), r.end(), [](auto& a, auto& b) { return a.first < b.first; });
        indptr[i + 1] = indptr[i] + (int)r.size();
    }
    int nnz = indptr[m];
    indices.resize(nnz);
    data.resize(nnz);
    for (int i = 0; i < m; ++i) {
        int p = indptr[i];
        for (auto& e : rows[i]) {
            indices[p] = e.first;
            data[p] = e.second;
            ++p;
        }
    }
}

}  // namespace

struct NrMpsProblem {
    int m, n, nnz;
    int* indptr;
    int* indices;
    double* data;
    double* c;
    double* lc;
    double* uc;
    double* lx;
    double* ux;
    double c0;
    double obj_sign;
    int* is_integer;
    int has_integer;
    int has_q;
    int q_nnz;
    int* q_indptr;
    int* q_indices;
    double* q_data;
    char* name;
    char* names_blob;   // m + n null-terminated strings concatenated
    int names_blob_len;
};

static char* dup_cstr(const std::string& s) {
    char* p = (char*)std::malloc(s.size() + 1);
    if (!p) return nullptr;
    std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

template <class T>
static T* dup_vec(const std::vector<T>& v) {
    if (v.empty()) return (T*)std::malloc(1);  // non-null empty
    T* p = (T*)std::malloc(sizeof(T) * v.size());
    if (p) std::memcpy(p, v.data(), sizeof(T) * v.size());
    return p;
}

NR_EXPORT void nr_mps_free(NrMpsProblem* p) {
    if (!p) return;
    std::free(p->indptr); std::free(p->indices); std::free(p->data);
    std::free(p->c); std::free(p->lc); std::free(p->uc); std::free(p->lx); std::free(p->ux);
    std::free(p->is_integer);
    std::free(p->q_indptr); std::free(p->q_indices); std::free(p->q_data);
    std::free(p->name); std::free(p->names_blob);
    std::memset(p, 0, sizeof(*p));
}

NR_EXPORT int nr_mps_parse(const char* text, int text_len, const char* default_name, int name_locked,
                           NrMpsProblem* out, char* errmsg, int errmsg_len) {
    if (!text || !out) return 1;
    std::memset(out, 0, sizeof(*out));
    ParseState st;
    st.prob_name = default_name ? default_name : "LP";
    st.name_locked = name_locked != 0;
    if (!parse_mps(st, text, text_len)) {
        if (errmsg && errmsg_len > 0) {
            std::snprintf(errmsg, errmsg_len, "%s", st.errmsg.c_str());
        }
        return 2;
    }
    int m = (int)st.row_names.size(), n = (int)st.col_names.size();
    std::vector<int> indptr, indices;
    std::vector<double> data;
    build_csr(st.trips, m, n, indptr, indices, data);

    std::vector<double> b(m, 0.0), lc(m), uc(m);
    for (auto& kv : st.rhs) b[kv.first] = kv.second;
    for (int i = 0; i < m; ++i) {
        const std::string& typ = st.row_types[i];
        if (typ == "L") { lc[i] = -kInf; uc[i] = b[i]; }
        else if (typ == "G") { lc[i] = b[i]; uc[i] = kInf; }
        else { lc[i] = b[i]; uc[i] = b[i]; }  // E
    }
    for (auto& kv : st.rng) {
        int i = kv.first;
        double r = kv.second;
        const std::string& typ = st.row_types[i];
        if (typ == "E") {
            if (r > 0) uc[i] = b[i] + std::fabs(r);
            else if (r < 0) lc[i] = b[i] - std::fabs(r);
        } else if (typ == "L") {
            lc[i] = b[i] - std::fabs(r);
        } else if (typ == "G") {
            uc[i] = b[i] + std::fabs(r);
        }
    }

    std::vector<double> lx(n, 0.0), ux(n, kInf);
    std::vector<char> lower_set(n, 0);
    std::vector<int> integer(n, 0);
    for (int j = 0; j < n; ++j) integer[j] = st.is_int[j] ? 1 : 0;

    for (auto& op : st.bnd_ops) {
        int j = op.j;
        double v = op.v;
        const std::string& typ = op.typ;
        if (typ == "UP" || typ == "UI") {
            ux[j] = v;
            if (v < 0 && !lower_set[j] && lx[j] == 0.0) {
                lx[j] = -kInf;
                st.notes.push_back("UP<0 on " + st.col_names[j] + ": lower bound set to -inf");
            }
            if (typ == "UI") integer[j] = 1;
        } else if (typ == "LO" || typ == "LI") {
            lx[j] = v; lower_set[j] = 1;
            if (typ == "LI") integer[j] = 1;
        } else if (typ == "FX") {
            lx[j] = ux[j] = v; lower_set[j] = 1;
        } else if (typ == "FR") {
            lx[j] = -kInf; ux[j] = kInf; lower_set[j] = 1;
        } else if (typ == "MI") {
            lx[j] = -kInf; lower_set[j] = 1;
        } else if (typ == "PL") {
            ux[j] = kInf;
        } else if (typ == "BV") {
            lx[j] = 0.0; ux[j] = 1.0; lower_set[j] = 1; integer[j] = 1;
        } else {
            if (errmsg && errmsg_len > 0)
                std::snprintf(errmsg, errmsg_len, "bound type %s not supported", typ.c_str());
            return 3;
        }
    }

    std::vector<double> c = st.cost;
    double c0 = st.c0, sense = st.sense;
    std::vector<Triplet> qtrips;
    for (auto& q : st.q_entries) {
        qtrips.push_back({q.i, q.j, q.v});
        if (q.mirror && q.i != q.j) qtrips.push_back({q.j, q.i, q.v});
    }
    std::vector<int> q_indptr, q_indices;
    std::vector<double> q_data;
    if (!qtrips.empty()) build_csr(qtrips, n, n, q_indptr, q_indices, q_data);

    if (sense < 0) {
        for (double& v : c) v = -v;
        c0 = -c0;
        for (double& v : q_data) v = -v;
    }

    int has_int = 0;
    for (int v : integer) if (v) { has_int = 1; break; }

    std::string blob;
    for (auto& s : st.row_names) { blob.append(s); blob.push_back('\0'); }
    for (auto& s : st.col_names) { blob.append(s); blob.push_back('\0'); }

    out->m = m; out->n = n; out->nnz = (int)data.size();
    out->indptr = dup_vec(indptr);
    out->indices = dup_vec(indices);
    out->data = dup_vec(data);
    out->c = dup_vec(c);
    out->lc = dup_vec(lc);
    out->uc = dup_vec(uc);
    out->lx = dup_vec(lx);
    out->ux = dup_vec(ux);
    out->c0 = c0;
    out->obj_sign = sense;
    out->is_integer = dup_vec(integer);
    out->has_integer = has_int;
    out->has_q = q_data.empty() ? 0 : 1;
    out->q_nnz = (int)q_data.size();
    out->q_indptr = out->has_q ? dup_vec(q_indptr) : nullptr;
    out->q_indices = out->has_q ? dup_vec(q_indices) : nullptr;
    out->q_data = out->has_q ? dup_vec(q_data) : nullptr;
    out->name = dup_cstr(st.prob_name);
    out->names_blob_len = (int)blob.size();
    out->names_blob = (char*)std::malloc(blob.size() ? blob.size() : 1);
    if (out->names_blob && !blob.empty()) std::memcpy(out->names_blob, blob.data(), blob.size());

    if (!out->indptr || !out->c || !out->name) {
        nr_mps_free(out);
        if (errmsg && errmsg_len > 0) std::snprintf(errmsg, errmsg_len, "out of memory");
        return 4;
    }
    return 0;
}
