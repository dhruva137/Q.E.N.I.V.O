#include "rational.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace qenivo {
namespace exact {

Rat::Rat(BigInt num, BigInt den) {
    if (den.is_zero()) {
        throw std::invalid_argument("zero denominator");
    }
    if (den.is_neg()) {
        num = -num;
        den = den.abs();
    }
    if (num.is_zero()) {
        n_ = BigInt(0);
        d_ = BigInt(1);
        return;
    }
    const BigInt g = binary_gcd(num, den);
    const auto qn = num.divmod(g);
    const auto qd = den.divmod(g);
    if (!qn.second.is_zero() || !qd.second.is_zero()) {
        throw std::logic_error("binary gcd did not divide the rational");
    }
    n_ = qn.first;
    d_ = qd.first;
}

Rat Rat::parse(const std::string& text) {
    const std::size_t slash = text.find('/');
    if (slash == std::string::npos) {
        return Rat(BigInt::from_dec(text), BigInt(1));
    }
    return Rat(BigInt::from_dec(text.substr(0, slash)), BigInt::from_dec(text.substr(slash + 1)));
}

Rat Rat::operator+(const Rat& other) const {
    return Rat(n_ * other.d_ + other.n_ * d_, d_ * other.d_);
}

Rat Rat::operator-(const Rat& other) const { return *this + (-other); }

Rat Rat::operator*(const Rat& other) const { return Rat(n_ * other.n_, d_ * other.d_); }

Rat Rat::operator/(const Rat& other) const { return Rat(n_ * other.d_, d_ * other.n_); }

double Rat::to_double() const {
    if (n_.is_zero()) {
        return 0.0;
    }
    const double num = std::stod(n_.to_dec());
    const double den = std::stod(d_.to_dec());
    return num / den;
}

std::string Rat::str() const {
    if (d_ == BigInt(1)) {
        return n_.to_dec();
    }
    return n_.to_dec() + "/" + d_.to_dec();
}

bool abs_greater(const Rat& a, const Rat& b) {
    const BigInt left = a.num().abs() * b.den();
    const BigInt right = b.num().abs() * a.den();
    return right < left;
}

Rat rat_from_double(double value) {
    if (value == 0.0) {
        return Rat(0);
    }
    if (!std::isfinite(value)) {
        throw std::invalid_argument("non-finite float");
    }
    std::uint64_t bits = 0;
    static_assert(sizeof(double) == 8, "IEEE-754 binary64 required");
    std::memcpy(&bits, &value, sizeof(bits));
    const bool neg = (bits >> 63) != 0;
    const int exp = static_cast<int>((bits >> 52) & 0x7ff);
    const std::uint64_t frac = bits & ((std::uint64_t{1} << 52) - 1);
    BigInt num;
    int shift = 0;
    if (exp == 0) {
        num = BigInt::from_u64(frac);
        shift = -1074;
    } else {
        num = BigInt::from_u64(frac | (std::uint64_t{1} << 52));
        shift = exp - 1075;
    }
    BigInt den(1);
    if (shift > 0) {
        num = num.shl(shift);
    } else if (shift < 0) {
        den = den.shl(-shift);
    }
    if (neg) {
        num = -num;
    }
    return Rat(std::move(num), std::move(den));
}

Rat limit_denominator(const Rat& value, const BigInt& max_den) {
    if (!max_den.is_neg() && value.den().cmp(max_den) <= 0) {
        return value;
    }
    const int sign = value.is_neg() ? -1 : 1;
    BigInt n = value.num().abs();
    BigInt d = value.den();
    BigInt p0(0);
    BigInt q0(1);
    BigInt p1(1);
    BigInt q1(0);
    while (!d.is_zero()) {
        const auto qr = n.divmod(d);
        const BigInt q2 = q0 + qr.first * q1;
        if (max_den < q2) {
            break;
        }
        const BigInt p2 = p0 + qr.first * p1;
        p0 = std::move(p1);
        q0 = std::move(q1);
        p1 = p2;
        q1 = q2;
        n = d;
        d = qr.second;
    }
    if (q1.is_zero()) {
        return Rat(0);
    }
    BigInt k(0);
    if (q0.cmp(max_den) <= 0 && !q1.is_zero()) {
        k = (max_den - q0).divmod(q1).first;
    }
    const Rat bound1(p0 + k * p1, q0 + k * q1);
    const Rat bound2(p1, q1);
    const auto gap = [&](const Rat& cand) {
        const BigInt prod = cand.num().abs() * value.den();
        const BigInt orig = value.num().abs() * cand.den();
        return prod.cmp(orig) >= 0 ? prod - orig : orig - prod;
    };
    const BigInt e1 = gap(bound1) * bound2.den();
    const BigInt e2 = gap(bound2) * bound1.den();
    Rat chosen = e2.cmp(e1) <= 0 ? bound2 : bound1;
    if (sign < 0) {
        chosen = -chosen;
    }
    return chosen;
}

}  // namespace exact
}  // namespace qenivo
