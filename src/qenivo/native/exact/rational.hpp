// Rationals in lowest terms. Reduction uses binary_gcd from bigint.hpp.
#ifndef QENIVO_EXACT_RATIONAL_HPP
#define QENIVO_EXACT_RATIONAL_HPP

#include "bigint.hpp"

#include <string>

namespace qenivo {
namespace exact {

class Rat {
public:
    Rat() : n_(0), d_(1) {}
    explicit Rat(std::int64_t num) : Rat(BigInt(num), BigInt(1)) {}
    Rat(std::int64_t num, std::int64_t den) : Rat(BigInt(num), BigInt(den)) {}
    Rat(BigInt num, BigInt den);

    static Rat parse(const std::string& text);

    const BigInt& num() const { return n_; }
    const BigInt& den() const { return d_; }
    bool is_zero() const { return n_.is_zero(); }
    bool is_neg() const { return n_.is_neg(); }
    bool is_pos() const { return !n_.is_neg() && !n_.is_zero(); }

    Rat operator-() const { return Rat(-n_, d_); }
    Rat operator+(const Rat& other) const;
    Rat operator-(const Rat& other) const;
    Rat operator*(const Rat& other) const;
    Rat operator/(const Rat& other) const;
    bool operator==(const Rat& other) const { return n_ == other.n_ && d_ == other.d_; }
    bool operator<(const Rat& other) const { return (*this - other).is_neg(); }
    Rat abs() const { return Rat(n_.abs(), d_); }
    double to_double() const;
    std::string str() const;

private:
    BigInt n_;
    BigInt d_;
};

bool abs_greater(const Rat& a, const Rat& b);
Rat rat_from_double(double value);
Rat limit_denominator(const Rat& value, const BigInt& max_den);

}  // namespace exact
}  // namespace qenivo

#endif
