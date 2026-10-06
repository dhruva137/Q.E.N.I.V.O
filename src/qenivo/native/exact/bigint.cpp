#include "bigint.hpp"

#include <algorithm>
#include <stdexcept>

namespace qenivo {
namespace exact {
namespace {

constexpr int kKaraBase = 4;

void add_into(std::vector<std::uint32_t>& dest, const std::vector<std::uint32_t>& src, std::size_t offset) {
    if (src.empty()) {
        return;
    }
    if (dest.size() < offset + src.size()) {
        dest.resize(offset + src.size(), 0);
    }
    std::uint64_t carry = 0;
    std::size_t i = 0;
    for (; i < src.size(); ++i) {
        const std::uint64_t t = static_cast<std::uint64_t>(dest[offset + i]) + src[i] + carry;
        dest[offset + i] = static_cast<std::uint32_t>(t);
        carry = t >> 32;
    }
    std::size_t k = offset + i;
    while (carry) {
        if (k == dest.size()) {
            dest.push_back(0);
        }
        const std::uint64_t t = static_cast<std::uint64_t>(dest[k]) + carry;
        dest[k] = static_cast<std::uint32_t>(t);
        carry = t >> 32;
        ++k;
    }
}

}  // namespace

void BigInt::norm(Limbs& limbs) {
    while (!limbs.empty() && limbs.back() == 0) {
        limbs.pop_back();
    }
}

BigInt BigInt::from_limbs(Limbs limbs, bool neg) {
    norm(limbs);
    BigInt out;
    out.d_ = std::move(limbs);
    out.neg_ = neg && !out.d_.empty();
    return out;
}

BigInt::BigInt(std::int64_t value) {
    std::uint64_t mag = 0;
    if (value < 0) {
        neg_ = true;
        mag = static_cast<std::uint64_t>(-(value + 1)) + 1u;
    } else {
        mag = static_cast<std::uint64_t>(value);
    }
    if (mag == 0) {
        neg_ = false;
        return;
    }
    d_.push_back(static_cast<std::uint32_t>(mag));
    const std::uint32_t hi = static_cast<std::uint32_t>(mag >> 32);
    if (hi != 0) {
        d_.push_back(hi);
    }
}

BigInt BigInt::from_u64(std::uint64_t value) {
    BigInt out;
    if (value == 0) {
        return out;
    }
    out.d_.push_back(static_cast<std::uint32_t>(value));
    const std::uint32_t hi = static_cast<std::uint32_t>(value >> 32);
    if (hi != 0) {
        out.d_.push_back(hi);
    }
    return out;
}

BigInt BigInt::from_dec(const std::string& text) {
    if (text.empty()) {
        throw std::invalid_argument("empty integer");
    }
    std::size_t i = 0;
    bool neg = false;
    if (text[i] == '+' || text[i] == '-') {
        neg = text[i] == '-';
        ++i;
    }
    if (i >= text.size()) {
        throw std::invalid_argument("integer has no digits");
    }
    BigInt out;
    for (; i < text.size(); ++i) {
        const char c = text[i];
        if (c < '0' || c > '9') {
            throw std::invalid_argument("bad integer digit");
        }
        out = out.mul_small(10) + BigInt(static_cast<std::int64_t>(c - '0'));
    }
    if (neg && !out.is_zero()) {
        out.neg_ = true;
    }
    return out;
}

int BigInt::bit_length() const {
    if (d_.empty()) {
        return 0;
    }
    const std::uint32_t top = d_.back();
    int bits = 0;
    for (std::uint32_t probe = top; probe != 0; probe >>= 1) {
        ++bits;
    }
    return static_cast<int>((d_.size() - 1) * 32) + bits;
}

int BigInt::cmp_limbs(const Limbs& a, const Limbs& b) {
    if (a.size() != b.size()) {
        return a.size() < b.size() ? -1 : 1;
    }
    for (std::size_t i = a.size(); i-- > 0;) {
        if (a[i] != b[i]) {
            return a[i] < b[i] ? -1 : 1;
        }
    }
    return 0;
}

int BigInt::cmp(const BigInt& other) const {
    if (neg_ != other.neg_) {
        return neg_ ? -1 : 1;
    }
    const int c = cmp_limbs(d_, other.d_);
    return neg_ ? -c : c;
}

BigInt BigInt::abs() const { return from_limbs(d_, false); }

BigInt BigInt::operator-() const { return from_limbs(d_, !neg_); }

BigInt::Limbs BigInt::add_limbs(const Limbs& a, const Limbs& b) {
    Limbs out;
    out.reserve(std::max(a.size(), b.size()) + 1);
    std::uint64_t carry = 0;
    const std::size_t n = std::max(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t av = i < a.size() ? a[i] : 0;
        const std::uint64_t bv = i < b.size() ? b[i] : 0;
        const std::uint64_t t = av + bv + carry;
        out.push_back(static_cast<std::uint32_t>(t));
        carry = t >> 32;
    }
    if (carry) {
        out.push_back(static_cast<std::uint32_t>(carry));
    }
    return out;
}

BigInt::Limbs BigInt::sub_limbs(const Limbs& a, const Limbs& b) {
    Limbs out(a.size(), 0);
    std::uint32_t borrow = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const std::uint64_t bv = i < b.size() ? b[i] : 0;
        std::uint64_t t = static_cast<std::uint64_t>(a[i]);
        if (t < static_cast<std::uint64_t>(bv) + borrow) {
            t += (static_cast<std::uint64_t>(1) << 32);
            out[i] = static_cast<std::uint32_t>(t - bv - borrow);
            borrow = 1;
        } else {
            out[i] = static_cast<std::uint32_t>(t - bv - borrow);
            borrow = 0;
        }
    }
    if (borrow) {
        throw std::logic_error("limb subtraction went negative");
    }
    norm(out);
    return out;
}

BigInt BigInt::operator+(const BigInt& other) const {
    if (neg_ == other.neg_) {
        return from_limbs(add_limbs(d_, other.d_), neg_);
    }
    const int c = cmp_limbs(d_, other.d_);
    if (c == 0) {
        return BigInt(0);
    }
    if (c > 0) {
        return from_limbs(sub_limbs(d_, other.d_), neg_);
    }
    return from_limbs(sub_limbs(other.d_, d_), other.neg_);
}

BigInt BigInt::operator-(const BigInt& other) const { return *this + (-other); }

BigInt::Limbs BigInt::school(const Limbs& a, const Limbs& b) {
    if (a.empty() || b.empty()) {
        return {};
    }
    Limbs out(a.size() + b.size(), 0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        std::uint64_t carry = 0;
        for (std::size_t j = 0; j < b.size(); ++j) {
            const std::uint64_t t = static_cast<std::uint64_t>(out[i + j]) +
                                    static_cast<std::uint64_t>(a[i]) * b[j] + carry;
            out[i + j] = static_cast<std::uint32_t>(t);
            carry = t >> 32;
        }
        std::size_t k = i + b.size();
        while (carry) {
            if (k == out.size()) {
                out.push_back(0);
            }
            const std::uint64_t t = static_cast<std::uint64_t>(out[k]) + carry;
            out[k] = static_cast<std::uint32_t>(t);
            carry = t >> 32;
            ++k;
        }
    }
    norm(out);
    return out;
}

BigInt::Limbs BigInt::karatsuba(const Limbs& x, const Limbs& y) {
    if (x.empty() || y.empty()) {
        return {};
    }
    if (x.size() < static_cast<std::size_t>(kKaraBase) || y.size() < static_cast<std::size_t>(kKaraBase)) {
        return school(x, y);
    }
    const std::size_t m = (std::max(x.size(), y.size()) + 1) / 2;
    const Limbs x0(x.begin(), x.begin() + std::min(m, x.size()));
    const Limbs x1 = x.size() > m ? Limbs(x.begin() + static_cast<std::ptrdiff_t>(m), x.end()) : Limbs{};
    const Limbs y0(y.begin(), y.begin() + std::min(m, y.size()));
    const Limbs y1 = y.size() > m ? Limbs(y.begin() + static_cast<std::ptrdiff_t>(m), y.end()) : Limbs{};
    const Limbs z0 = karatsuba(x0, y0);
    const Limbs z2 = karatsuba(x1, y1);
    Limbs z1 = karatsuba(add_limbs(x0, x1), add_limbs(y0, y1));
    z1 = sub_limbs(z1, z0);
    z1 = sub_limbs(z1, z2);
    Limbs out = z0;
    add_into(out, z1, m);
    add_into(out, z2, 2 * m);
    norm(out);
    return out;
}

BigInt BigInt::mul_schoolbook(const BigInt& other) const {
    if (is_zero() || other.is_zero()) {
        return BigInt(0);
    }
    return from_limbs(school(d_, other.d_), neg_ != other.neg_);
}

BigInt BigInt::mul_karatsuba(const BigInt& other) const {
    if (is_zero() || other.is_zero()) {
        return BigInt(0);
    }
    return from_limbs(karatsuba(d_, other.d_), neg_ != other.neg_);
}

BigInt BigInt::operator*(const BigInt& other) const {
    if (d_.size() >= kCutoff && other.d_.size() >= kCutoff) {
        return mul_karatsuba(other);
    }
    return mul_schoolbook(other);
}

BigInt BigInt::shl(int bits) const {
    if (bits < 0) {
        throw std::invalid_argument("negative shift");
    }
    if (bits == 0 || is_zero()) {
        return from_limbs(d_, neg_);
    }
    const int limb_shift = bits >> 5;
    const int bit_shift = bits & 31;
    Limbs out(static_cast<std::size_t>(limb_shift), 0);
    std::uint32_t carry = 0;
    for (std::uint32_t limb : d_) {
        const std::uint64_t cur = (static_cast<std::uint64_t>(limb) << bit_shift) + carry;
        out.push_back(static_cast<std::uint32_t>(cur));
        carry = static_cast<std::uint32_t>(cur >> 32);
    }
    if (carry) {
        out.push_back(carry);
    }
    return from_limbs(std::move(out), neg_);
}

BigInt BigInt::shr(int bits) const {
    if (bits < 0) {
        throw std::invalid_argument("negative shift");
    }
    if (bits == 0 || is_zero()) {
        return from_limbs(d_, neg_);
    }
    const std::size_t limb_shift = static_cast<std::size_t>(bits >> 5);
    const int bit_shift = bits & 31;
    if (limb_shift >= d_.size()) {
        return BigInt(0);
    }
    Limbs src(d_.begin() + static_cast<std::ptrdiff_t>(limb_shift), d_.end());
    if (bit_shift == 0) {
        return from_limbs(std::move(src), neg_);
    }
    Limbs out;
    out.reserve(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        std::uint32_t cur = src[i] >> bit_shift;
        if (i + 1 < src.size()) {
            cur |= (src[i + 1] & ((1u << bit_shift) - 1u)) << (32 - bit_shift);
        }
        out.push_back(cur);
    }
    return from_limbs(std::move(out), neg_);
}

BigInt BigInt::mul_small(std::uint32_t k) const {
    if (k == 0 || is_zero()) {
        return BigInt(0);
    }
    Limbs out;
    out.reserve(d_.size() + 1);
    std::uint64_t carry = 0;
    for (std::uint32_t limb : d_) {
        const std::uint64_t t = static_cast<std::uint64_t>(limb) * k + carry;
        out.push_back(static_cast<std::uint32_t>(t));
        carry = t >> 32;
    }
    if (carry) {
        out.push_back(static_cast<std::uint32_t>(carry));
    }
    return from_limbs(std::move(out), neg_);
}

std::pair<BigInt, BigInt> BigInt::divmod_pos(const BigInt& a, const BigInt& b) {
    if (a.cmp(b) < 0) {
        return {BigInt(0), a};
    }
    const int shift = a.bit_length() - b.bit_length();
    BigInt r = a;
    BigInt q(0);
    BigInt bshift = b.shl(shift);
    for (int i = shift; i >= 0; --i) {
        if (r.cmp(bshift) >= 0) {
            r = r - bshift;
            Limbs bits = q.d_;
            const std::size_t limb = static_cast<std::size_t>(i >> 5);
            const std::uint32_t bit = 1u << (i & 31);
            if (bits.size() <= limb) {
                bits.resize(limb + 1, 0);
            }
            bits[limb] |= bit;
            q = from_limbs(std::move(bits), false);
        }
        bshift = bshift.shr(1);
    }
    return {q, r};
}

std::pair<BigInt, BigInt> BigInt::divmod(const BigInt& other) const {
    if (other.is_zero()) {
        throw std::invalid_argument("division by zero");
    }
    auto [q, r] = divmod_pos(abs(), other.abs());
    if (neg_ != other.neg_) {
        q = -q;
    }
    if (neg_ && !r.is_zero()) {
        r = -r;
    }
    return {q, r};
}

std::string BigInt::to_dec() const {
    if (is_zero()) {
        return "0";
    }
    BigInt mag = abs();
    std::string digits;
    while (!mag.is_zero()) {
        std::uint64_t rem = 0;
        for (std::size_t i = mag.d_.size(); i-- > 0;) {
            const std::uint64_t cur = (rem << 32) | mag.d_[i];
            mag.d_[i] = static_cast<std::uint32_t>(cur / 10);
            rem = cur % 10;
        }
        norm(mag.d_);
        digits.push_back(static_cast<char>('0' + rem));
    }
    if (neg_) {
        digits.push_back('-');
    }
    std::reverse(digits.begin(), digits.end());
    return digits;
}

BigInt binary_gcd(BigInt a, BigInt b) {
    a = a.abs();
    b = b.abs();
    if (a.is_zero()) {
        return b;
    }
    if (b.is_zero()) {
        return a;
    }
    int shift = 0;
    while (a.is_even() && b.is_even()) {
        a = a.shr(1);
        b = b.shr(1);
        ++shift;
    }
    while (a.is_even()) {
        a = a.shr(1);
    }
    while (!b.is_zero()) {
        while (b.is_even()) {
            b = b.shr(1);
        }
        if (a.cmp(b) > 0) {
            std::swap(a, b);
        }
        b = b - a;
    }
    return shift ? a.shl(shift) : a;
}

}  // namespace exact
}  // namespace qenivo
