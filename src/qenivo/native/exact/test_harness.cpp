// Tiny driver for the exact-arithmetic core. No GoogleTest.
#include "lu.hpp"
#include "refine.hpp"

#include <iostream>
#include <string>

namespace {

int g_failed = 0;

void expect(bool ok, const std::string& name) {
    if (ok) {
        std::cout << "PASS " << name << "\n";
    } else {
        std::cout << "FAIL " << name << "\n";
        ++g_failed;
    }
}

qenivo::exact::BigInt dec(const std::string& text) { return qenivo::exact::BigInt::from_dec(text); }

std::string nines(int count) { return std::string(static_cast<std::size_t>(count), '9'); }

std::string sevens(int count) { return std::string(static_cast<std::size_t>(count), '7'); }

}  // namespace

int main() {
    using qenivo::exact::BigInt;
    using qenivo::exact::Rat;
    using qenivo::exact::binary_gcd;
    using qenivo::exact::solve_exact;
    using qenivo::exact::solve_refined;
    using qenivo::exact::solve_system;

    const Rat third(2, 3);
    const Rat sum = (third + third) + third;
    expect(sum == Rat(2, 1) && sum.str() == "2", "2/3 + 2/3 + 2/3 = 2");
    expect(Rat(4, 6) == Rat(2, 3) && Rat(4, 6).str() == "2/3", "4/6 reduces to 2/3");
    expect(binary_gcd(BigInt(84), BigInt(30)) == BigInt(6), "gcd(84, 30) = 6");

    const BigInt a = dec(nines(90));
    const BigInt b = dec(sevens(80));
    expect(a.mul_schoolbook(b) == a.mul_karatsuba(b), "karatsuba matches schoolbook on 90-digit integers");
    const BigInt neg = -dec(nines(40));
    expect(neg.mul_schoolbook(b) == neg.mul_karatsuba(b), "karatsuba matches schoolbook on a negative integer");

    BigInt ten(1);
    for (int i = 0; i < 80; ++i) {
        ten = ten.mul_small(10);
    }
    const BigInt ident = (ten - BigInt(1)).mul_karatsuba(ten + BigInt(1));
    BigInt ten160(1);
    for (int i = 0; i < 160; ++i) {
        ten160 = ten160.mul_small(10);
    }
    expect(ident == ten160 - BigInt(1), "(10^80-1)*(10^80+1) = 10^160-1");

    const std::vector<std::vector<Rat>> basis{{Rat(1), Rat(2)}, {Rat(2), Rat(1)}};
    const std::vector<Rat> rhs{Rat(4), Rat(4)};
    const std::vector<Rat> lu = solve_exact(basis, rhs);
    expect(lu.size() == 2 && lu[0] == Rat(4, 3) && lu[1] == Rat(4, 3), "exact LU solves the 2x2 basis as 4/3");
    const auto refined = solve_refined(basis, rhs);
    expect(refined.has_value() && (*refined)[0] == Rat(4, 3) && (*refined)[1] == Rat(4, 3),
           "iterative refinement recovers 4/3");
    const auto solved = solve_system(basis, rhs);
    expect(solved.has_value() && solved->method == "iterative_refinement" && solved->x[0] == Rat(4, 3),
           "basis solve prefers iterative refinement");

    if (g_failed != 0) {
        std::cout << g_failed << " failed\n";
        return 1;
    }
    std::cout << "all passed\n";
    return 0;
}
