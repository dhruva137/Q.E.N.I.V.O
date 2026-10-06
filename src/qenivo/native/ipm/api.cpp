// C ABI used by engines/native_ipm.py. The handle owns one supernodal LDL' factor.

#include "ipm_internal.hpp"

#include <algorithm>
#include <cstdlib>
#include <vector>

extern "C" {

int nr_ipm_order(int n, const int* colptr, const int* rowidx, int method, int* perm) {
    if (n < 0 || perm == nullptr) {
        return 1;
    }
    if (n == 0) {
        return 0;
    }
    std::vector<std::vector<int>> adj = qenivo::adjacency(n, colptr, rowidx);
    std::vector<int> order = method == 0 ? qenivo::amd_from_adj(adj) : qenivo::nd_from_adj(adj);
    if (!qenivo::is_permutation(order, n)) {
        for (int i = 0; i < n; ++i) {
            perm[i] = i;
        }
        return 2;
    }
    for (int i = 0; i < n; ++i) {
        perm[i] = order[i];
    }
    return 0;
}

int nr_ipm_symbolic(int n, const int* colptr, const int* rowidx, const int* perm, int* parent, int* colcount,
                    int* sn_id, int* nsnodes) {
    if (n < 0) {
        return 1;
    }
    std::vector<int> elimination(n);
    if (perm == nullptr) {
        for (int i = 0; i < n; ++i) {
            elimination[i] = i;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            elimination[i] = perm[i];
        }
    }
    qenivo::Assembled lower = qenivo::assemble_lower(n, colptr, rowidx, nullptr, elimination.data());
    qenivo::Symbolic sym = qenivo::symbolic_factor(lower);
    if (parent != nullptr) {
        for (int i = 0; i < n; ++i) {
            parent[i] = sym.parent[i];
        }
    }
    if (colcount != nullptr) {
        for (int i = 0; i < n; ++i) {
            colcount[i] = sym.colcount[i];
        }
    }
    if (sn_id != nullptr) {
        for (int i = 0; i < n; ++i) {
            sn_id[i] = sym.sn_id[i];
        }
    }
    if (nsnodes != nullptr) {
        *nsnodes = n == 0 ? 0 : (int)sym.sn_start.size() - 1;
    }
    return 0;
}

void* nr_ipm_factor(int n, const int* colptr, const int* rowidx, const double* values, const int* perm, int spd,
                    double reg, int threads, int* nneg, double* reg_used) {
    auto* factor = new qenivo::Factor();
    std::vector<int> elimination(std::max(n, 0));
    if (perm == nullptr) {
        for (int i = 0; i < n; ++i) {
            elimination[i] = i;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            elimination[i] = perm[i];
        }
    }
    if (!factor->run(n, colptr, rowidx, values, elimination, spd, reg, threads)) {
        delete factor;
        return nullptr;
    }
    if (nneg != nullptr) {
        *nneg = factor->nneg;
    }
    if (reg_used != nullptr) {
        *reg_used = factor->reg_added;
    }
    return factor;
}

int nr_ipm_solve(void* handle, const double* b, double* x) {
    if (handle == nullptr || b == nullptr || x == nullptr) {
        return 1;
    }
    static_cast<qenivo::Factor*>(handle)->solve(b, x);
    return 0;
}

int nr_ipm_nneg(void* handle) {
    return handle == nullptr ? 0 : static_cast<qenivo::Factor*>(handle)->nneg;
}

double nr_ipm_regularisation(void* handle) {
    return handle == nullptr ? 0.0 : static_cast<qenivo::Factor*>(handle)->reg_added;
}

int nr_ipm_threads(void* handle) {
    return handle == nullptr ? 0 : static_cast<qenivo::Factor*>(handle)->threads_used;
}

void nr_ipm_free(void* handle) {
    delete static_cast<qenivo::Factor*>(handle);
}

}  // extern "C"
