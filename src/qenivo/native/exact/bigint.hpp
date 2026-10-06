// Arbitrary-precision integers for the exact LP engine.
//
// Schoolbook multiplication, Karatsuba above a small limb cutoff
// (A. Karatsuba and Yu. Ofman, Doklady Akad. Nauk SSSR, 1962), and Stein's
// binary GCD (J. Stein, Journal of Computational Physics, 1967).
// No GMP and no Boost.Multiprecision.
#ifndef QENIVO_EXACT_BIGINT_HPP
#define QENIVO_EXACT_BIGINT_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace qenivo {
namespace exact {

class BigInt {
public:
    BigInt() = default;
    explicit BigInt(std::int64_t value);
    static BigInt from_u64(std::uint64_t value);
    static BigInt from_dec(const std::string& text);

    bool is_zero() const { return d_.empty(); }
    bool is_neg() const { return neg_; }
    bool is_even() const { return d_.empty() || (d_[0] & 1u) == 0u; }
    int sign() const { return is_zero() ? 0 : (neg_ ? -1 : 1); }
    std::size_t limbs() const { return d_.size(); }
    int bit_length() const;

    int cmp(const BigInt& other) const;
    bool operator==(const BigInt& other) const { return cmp(other) == 0; }
    bool operator<(const BigInt& other) const { return cmp(other) < 0; }

    BigInt abs() const;
    BigInt operator-() const;
    BigInt operator+(const BigInt& other) const;
    BigInt operator-(const BigInt& other) const;
    BigInt operator*(const BigInt& other) const;
    BigInt mul_schoolbook(const BigInt& other) const;
    BigInt mul_karatsuba(const BigInt& other) const;
    BigInt shl(int bits) const;
    BigInt shr(int bits) const;
    // Quotient and remainder. Remainder is zero when `other` divides `*this`.
    std::pair<BigInt, BigInt> divmod(const BigInt& other) const;
    BigInt mul_small(std::uint32_t k) const;
    std::string to_dec() const;

    static constexpr std::size_t kCutoff = 8;

private:
    using Limbs = std::vector<std::uint32_t>;
    bool neg_ = false;
    Limbs d_;

    static BigInt from_limbs(Limbs limbs, bool neg);
    static Limbs add_limbs(const Limbs& a, const Limbs& b);
    static Limbs sub_limbs(const Limbs& a, const Limbs& b);
    static int cmp_limbs(const Limbs& a, const Limbs& b);
    static Limbs school(const Limbs& a, const Limbs& b);
    static Limbs karatsuba(const Limbs& a, const Limbs& b);
    static void norm(Limbs& limbs);
    static std::pair<BigInt, BigInt> divmod_pos(const BigInt& a, const BigInt& b);
};

BigInt binary_gcd(BigInt a, BigInt b);

}  // namespace exact
}  // namespace qenivo

#endif
