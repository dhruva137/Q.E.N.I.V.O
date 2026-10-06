// C ABI used by engines/exact_lp.py. Strings are malloc'd; the caller frees them
// with nr_free. A null return means the text could not be parsed or the matrix
// was singular.
#include "refine.hpp"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#if defined(_WIN32)
#define NR_API __declspec(dllexport)
#else
#define NR_API __attribute__((visibility("default")))
#endif

namespace {

char* dup(const std::string& text) {
    char* out = static_cast<char*>(std::malloc(text.size() + 1));
    if (out == nullptr) {
        return nullptr;
    }
    std::memcpy(out, text.c_str(), text.size() + 1);
    return out;
}

std::vector<std::string> split(const std::string& text, char sep) {
    std::vector<std::string> parts;
    std::string cur;
    for (char c : text) {
        if (c == sep) {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    parts.push_back(cur);
    return parts;
}

}  // namespace

extern "C" {

NR_API char* nr_q_add(const char* a, const char* b) {
    if (a == nullptr || b == nullptr) {
        return nullptr;
    }
    try {
        const qenivo::exact::Rat sum = qenivo::exact::Rat::parse(a) + qenivo::exact::Rat::parse(b);
        return dup(sum.str());
    } catch (const std::exception&) {
        return nullptr;
    }
}

NR_API char* nr_mul(const char* a, const char* b, int mode) {
    if (a == nullptr || b == nullptr) {
        return nullptr;
    }
    try {
        const qenivo::exact::BigInt left = qenivo::exact::BigInt::from_dec(a);
        const qenivo::exact::BigInt right = qenivo::exact::BigInt::from_dec(b);
        qenivo::exact::BigInt prod(0);
        if (mode == 1) {
            prod = left.mul_schoolbook(right);
        } else if (mode == 2) {
            prod = left.mul_karatsuba(right);
        } else {
            prod = left * right;
        }
        return dup(prod.to_dec());
    } catch (const std::exception&) {
        return nullptr;
    }
}

NR_API char* nr_gcd(const char* a, const char* b) {
    if (a == nullptr || b == nullptr) {
        return nullptr;
    }
    try {
        const qenivo::exact::BigInt g =
            qenivo::exact::binary_gcd(qenivo::exact::BigInt::from_dec(a), qenivo::exact::BigInt::from_dec(b));
        return dup(g.to_dec());
    } catch (const std::exception&) {
        return nullptr;
    }
}

NR_API char* nr_solve(const char* basis, const char* rhs) {
    if (basis == nullptr || rhs == nullptr) {
        return nullptr;
    }
    try {
        const std::vector<std::string> row_text = split(basis, ';');
        std::vector<std::vector<qenivo::exact::Rat>> matrix;
        matrix.reserve(row_text.size());
        for (const std::string& row : row_text) {
            if (row.empty()) {
                continue;
            }
            std::vector<qenivo::exact::Rat> parsed;
            for (const std::string& entry : split(row, ',')) {
                parsed.push_back(qenivo::exact::Rat::parse(entry));
            }
            matrix.push_back(std::move(parsed));
        }
        std::vector<qenivo::exact::Rat> right;
        for (const std::string& entry : split(rhs, ',')) {
            if (!entry.empty()) {
                right.push_back(qenivo::exact::Rat::parse(entry));
            }
        }
        const auto solved = qenivo::exact::solve_system(matrix, right);
        if (!solved) {
            return nullptr;
        }
        std::string body = solved->method;
        body.push_back('\n');
        for (std::size_t i = 0; i < solved->x.size(); ++i) {
            if (i) {
                body.push_back(',');
            }
            body += solved->x[i].str();
        }
        return dup(body);
    } catch (const std::exception&) {
        return nullptr;
    }
}

NR_API void nr_free(char* p) { std::free(p); }

}
