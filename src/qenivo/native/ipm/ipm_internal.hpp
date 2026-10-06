#ifndef NOMINMAX
#define NOMINMAX
#endif
#pragma once

// Shared declarations for the native sparse LDL' kernel.
//
// Ordering: Amestoy, Davis, Duff, SIAM J. Matrix Anal. Appl. 17 (1996);
//   George, SIAM J. Numer. Anal. 10 (1973); Kernighan and Lin, Bell Syst. Tech. J. 49 (1970);
//   Fiduccia and Mattheyses, DAC 1982; Karypis and Kumar, SIAM J. Sci. Comput. 20 (1998).
// Symbolic: Liu, SIAM J. Matrix Anal. Appl. 11 (1990);
//   Rose, Tarjan and Lueker, SIAM J. Comput. 5 (1976);
//   Liu, Ng and Peyton, SIAM J. Matrix Anal. Appl. 14 (1993).
// Numeric: Ng and Peyton, SIAM J. Sci. Comput. 14 (1993);
//   Altman and Gondzio, Optim. Methods Softw. 11 (1999);
//   Friedlander and Orban, Math. Program. Comput. 4 (2012).

#include <vector>

namespace qenivo {

struct Assembled {
    int n = 0;
    std::vector<int> colptr;
    std::vector<int> rowidx;
    std::vector<double> values;
    std::vector<double> diag;
    std::vector<std::vector<int>> strict;
    double scale = 1.0;
};

struct Symbolic {
    int n = 0;
    std::vector<int> parent;
    std::vector<int> colcount;
    std::vector<std::vector<int>> pattern;
    std::vector<int> sn_id;
    std::vector<int> sn_start;
    std::vector<int> sn_parent;
};

std::vector<std::vector<int>> adjacency(int n, const int* colptr, const int* rowidx);

std::vector<int> amd_from_adj(const std::vector<std::vector<int>>& adj);

std::vector<int> nd_from_adj(const std::vector<std::vector<int>>& adj);

bool is_permutation(const std::vector<int>& perm, int n);

Assembled assemble_lower(int n, const int* colptr, const int* rowidx, const double* values, const int* perm);

Symbolic symbolic_factor(const Assembled& lower);

struct Factor {
    int n = 0;
    int nneg = 0;
    int threads_used = 1;
    double reg_added = 0.0;
    std::vector<int> perm;
    std::vector<int> inv;
    std::vector<int> sp_colptr;
    std::vector<int> sp_rowidx;
    std::vector<double> sp_values;

    struct Supernode {
        int c0 = 0;
        int c1 = 0;
        std::vector<int> idx;
        std::vector<double> L;
        std::vector<double> D;
        std::vector<int> contrib;
        int nneg = 0;
        double reg_added = 0.0;
    };
    std::vector<Supernode> sn;
    std::vector<int> sn_parent;

    bool run(int n, const int* colptr, const int* rowidx, const double* values, const std::vector<int>& elimination,
             int spd, double reg, int threads);
    void solve(const double* b, double* x) const;
};

}  // namespace qenivo

extern "C" {
int nr_ipm_order(int n, const int* colptr, const int* rowidx, int method, int* perm);
int nr_ipm_symbolic(int n, const int* colptr, const int* rowidx, const int* perm, int* parent, int* colcount, int* sn_id,
                    int* nsnodes);
void* nr_ipm_factor(int n, const int* colptr, const int* rowidx, const double* values, const int* perm, int spd, double reg,
                    int threads, int* nneg, double* reg_used);
int nr_ipm_solve(void* handle, const double* b, double* x);
int nr_ipm_nneg(void* handle);
double nr_ipm_regularisation(void* handle);
int nr_ipm_threads(void* handle);
void nr_ipm_free(void* handle);
}
