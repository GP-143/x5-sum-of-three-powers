// x^5 + y^5 + z^5 = n  for every 0 < n < 100,  |x|,|y|,|z| <= RANGE          single file, C++17, OpenMP
//
//   g++ -O3 -march=native -fopenmp solve.cpp -o solve
//   ./solve --selftest              brute-force cross-check on a small range; run this first
//   ./solve --test                  unit tests (fifth roots, 256-bit arithmetic)
//   ./solve --run 9999999 out.txt   the search; the sieve needs about 0.8 GB per 10^8 of RANGE
//
// Why not brute force or meet-in-the-middle: three variables do not split into two equal halves, so hashing
// z^5 and looping over (x,y) is still ~RANGE^2 pairs.  This search is driven by an algebraic identity instead.
//
// Identity.  With s = x+y and p = xy:   x^5 + y^5 = s^5 - 5 s^3 p + 5 s p^2 = s (s^4 - 5 s^2 p + 5 p^2).
// For fixed z put m = n - z^5 = x^5 + y^5.  Then s | m, i.e.  z^5 == n (mod s).  For a given s this congruence is
// cheap to solve (factor s, take fifth roots modulo each prime power, combine by CRT).  Given (s, z):
//     5 s p^2 - 5 s^3 p + (s^5 - m) = 0      needs   Disc = 5 s (s^5 + 4 m)   to be a perfect square,
// and then x,y are the roots of t^2 - s t + p, which needs s^2 - 4p to be a perfect square.
//
// Size pruning.  Since n < 100 the three fifth powers nearly cancel.  If every |variable| exceeds A0
// (A0^5 <= n_max < (A0+1)^5) then the two smaller-magnitude variables have the same sign and the largest, z, the
// opposite one.  With S = |x+y| and Z = |z|, the power-mean inequality S^5/16 <= x^5+y^5 <= S^5 puts Z in
//     [ 2^(-4/5) S - 1 , S ],
// a window shorter than one period of the modulus S, so each (S, n, root of Z^5 == +-n mod S) gives at most one
// candidate, and S only has to run up to 2^(4/5) RANGE.  Solutions with a tiny variable lie in a small box that
// is enumerated directly (small_solutions).
//
// Big integers.  Z^5 reaches 10^40 and Disc about 10^50, beyond __int128 (1.7e38).  Z256 below is a small
// purpose-built 256-bit layer; its multiplications assert that nothing overflows, so do not compile with -DNDEBUG.
//
// Infinite families.  If n = c^5 (n = 1 with c = 1, n = 32 with c = 2) then (a, -a, c) solves the equation for
// every integer a, in any slot.  These are reported, never enumerated, and filtered out of the results.
//
// Bug log (found while building this; the tests keep them fixed)
//   1. crt_combine: operator-precedence slip in the modular difference gave wrong residues for composite S.
//   2. isqrt through long double only: exact below 2^128, but at RANGE = 10^8 the estimate is off by ~10^5, so the
//      +-1 correction loop would run ~10^5 times per call.  Now one exact integer Newton step; --test checks
//      roots up to 2^127 against the bit-by-bit reference.
//   3. A per-candidate atomic counter would serialise the threads; counts are now accumulated per S.
//
// Limits: RANGE below about 2.4e9 (32-bit sieve entries, v^4 < 2^128); the sieve is the real limit, 4 bytes per
// value of S up to 1.74*RANGE.  n_max must fit an int.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

using u64  = uint64_t;
using u32  = uint32_t;
using i64  = int64_t;
using u128 = unsigned __int128;
using i128 = __int128;

// 256-bit integers
struct U256 { u64 limb[4] = {0,0,0,0}; };

static inline U256 u256_from_u128(u128 v){ U256 r; r.limb[0]=(u64)v; r.limb[1]=(u64)(v>>64); return r; }
static inline int  u256_cmp(const U256&a,const U256&b){ for(int i=3;i>=0;--i) if(a.limb[i]!=b.limb[i]) return a.limb[i]<b.limb[i]?-1:1; return 0; }
static inline bool u256_is_zero(const U256&a){ return (a.limb[0]|a.limb[1]|a.limb[2]|a.limb[3])==0; }

static inline U256 u256_add(const U256&a,const U256&b){
    U256 r; u128 carry=0;
    for(int i=0;i<4;i++){ u128 s=(u128)a.limb[i]+b.limb[i]+carry; r.limb[i]=(u64)s; carry=s>>64; }
    assert(carry==0 && "256-bit add overflow: magnitude bound assumption violated");
    return r;
}
static inline U256 u256_sub(const U256&a,const U256&b){ // requires a>=b
    U256 r; i128 borrow=0;
    for(int i=0;i<4;i++){ i128 d=(i128)a.limb[i]-b.limb[i]-borrow; if(d<0){d+=((i128)1<<64); borrow=1;} else borrow=0; r.limb[i]=(u64)d; }
    assert(borrow==0);
    return r;
}
static inline U256 mul128_to_256(u128 a,u128 b){
    u64 a0=(u64)a,a1=(u64)(a>>64), b0=(u64)b,b1=(u64)(b>>64);
    u128 p00=(u128)a0*b0, p01=(u128)a0*b1, p10=(u128)a1*b0, p11=(u128)a1*b1;
    u128 acc1 = (p00>>64) + (u64)p01 + (u64)p10;
    u128 acc2 = (acc1>>64) + (p01>>64) + (p10>>64) + (u64)p11;
    u128 acc3 = (acc2>>64) + (p11>>64);
    U256 r; r.limb[0]=(u64)p00; r.limb[1]=(u64)acc1; r.limb[2]=(u64)acc2; r.limb[3]=(u64)acc3;
    assert((acc3>>64)==0 && "256-bit multiply overflow");
    return r;
}
static inline U256 u256_mul_small(const U256&a, u64 s){
    U256 r; u128 carry=0;
    for(int i=0;i<4;i++){ u128 p=(u128)a.limb[i]*s+carry; r.limb[i]=(u64)p; carry=p>>64; }
    assert(carry==0);
    return r;
}
// Bit-by-bit reference isqrt: exact at any magnitude, but ~128 multiplications; used by --test and as a fallback.
static inline u128 isqrt_u256_reference(const U256&n, bool* exact){
    u128 res=0;
    for(int i=127;i>=0;--i){ u128 cand=res|((u128)1<<i); if(u256_cmp(mul128_to_256(cand,cand), n)<=0) res=cand; }
    *exact = (u256_cmp(mul128_to_256(res,res), n)==0);
    return res;
}

// Fast isqrt.  long double (64-bit mantissa) estimates sqrt(n) only to ~2^-63 relative accuracy, which for the
// ~2^84-sized roots that occur at RANGE=10^8 means an absolute error of ~10^5 -- far too many +-1 correction
// steps.  So: take the estimate, do ONE exact integer Newton step using the exact residual n - r^2 (which fits
// in 128 bits), leaving an error ~10^-14, then finish with the usual +-1 corrections.
// (An earlier version without the Newton step was only validated for n < 2^128, where the estimate is within
// 1 of the truth; --test now checks roots up to 2^127 against the bit-by-bit reference.)
static inline long double u256_to_long_double(const U256& n){
    long double result = 0.0L;
    for (int i=3;i>=0;i--) result = result * 18446744073709551616.0L + (long double)n.limb[i];
    return result;
}
static inline u128 isqrt_u256(const U256& n, bool* exact){
    if (u256_is_zero(n)) { *exact=true; return 0; }
    long double rd = sqrtl(u256_to_long_double(n));
    u128 r = (rd >= 1.7e38L) ? (((u128)1)<<127) : (u128)rd;
    if (r == 0) r = 1;
    U256 sq = mul128_to_256(r, r);
    int c = u256_cmp(sq, n);
    if (c != 0){
        U256 diff = (c > 0) ? u256_sub(sq, n) : u256_sub(n, sq);
        if (diff.limb[2]==0 && diff.limb[3]==0){
            u128 e = ((u128)diff.limb[1] << 64) | diff.limb[0];
            u128 delta = (e / r) / 2;                       // Newton: r' = r -/+ |r^2-n| / (2r)
            r = (c > 0) ? r - delta : r + delta;
        } else {
            return isqrt_u256_reference(n, exact);          // residual > 2^128: only for n near 2^256, never in this program
        }
    }
    while (r > 0 && u256_cmp(mul128_to_256(r,r), n) > 0) r--;
    while (u256_cmp(mul128_to_256(r+1,r+1), n) <= 0) r++;
    *exact = (u256_cmp(mul128_to_256(r,r), n)==0);
    return r;
}

struct Z256 { bool neg=false; U256 mag; };
static inline Z256 z256_from_i64(i64 v){ Z256 r; r.neg=v<0; u64 m=r.neg?(u64)(-(i128)v):(u64)v; r.mag=u256_from_u128(m); return r; }
static inline Z256 z256_neg(Z256 a){ if(!u256_is_zero(a.mag)) a.neg=!a.neg; return a; }
static inline Z256 z256_add(const Z256&a,const Z256&b){
    Z256 r;
    if (a.neg==b.neg){ r.neg=a.neg; r.mag=u256_add(a.mag,b.mag); }
    else { int c=u256_cmp(a.mag,b.mag); if (c>=0){ r.mag=u256_sub(a.mag,b.mag); r.neg=a.neg; } else { r.mag=u256_sub(b.mag,a.mag); r.neg=b.neg; } }
    if (u256_is_zero(r.mag)) r.neg=false;
    return r;
}
static inline Z256 z256_sub(const Z256&a,const Z256&b){ return z256_add(a, z256_neg(b)); }
static inline Z256 z256_mul_small(const Z256&a, i64 s){
    Z256 r; u64 sm = s<0?(u64)(-(i128)s):(u64)s;
    r.mag = u256_mul_small(a.mag, sm); r.neg = (a.neg != (s<0));
    if (u256_is_zero(r.mag)) r.neg=false;
    return r;
}
static inline Z256 pow5_Z256(i64 v){
    u128 av=(u128)(v<0?(i128)(-(i128)v):(i128)v);
    u128 v2=av*av, v4=v2*v2;
    U256 mag = mul128_to_256(v4, av);
    Z256 r; r.mag=mag; r.neg=(v<0); if(u256_is_zero(r.mag)) r.neg=false;
    return r;
}

// modular helpers
static inline u64 mulmod(u64 a,u64 b,u64 m){ return (m <= 0xFFFFFFFFULL) ? (a*b)%m : (u64)(((u128)a*b)%m); }
static inline u64 powmod(u64 base,u64 e,u64 m){ base%=m; u64 r=1%m; while(e){ if(e&1) r=mulmod(r,base,m); base=mulmod(base,base,m); e>>=1; } return r; }
static inline i64 egcd(i64 a,i64 b,i64&x,i64&y){ if(b==0){x=1;y=0;return a;} i64 x1,y1; i64 g=egcd(b,a%b,x1,y1); x=y1; y=x1-(a/b)*y1; return g; }
static inline u64 modinv(u64 a,u64 m){ i64 x,y; egcd((i64)(a%m),(i64)m,x,y); i64 r=x%(i64)m; if(r<0) r+=(i64)m; return (u64)r; }

// Per-prime data that does not depend on the target residue, computed once per (S, prime factor).
struct PrimeInfo {
    u64 p = 0;
    int kind = 0;               // 0: 5th-power map is a bijection mod p;  1: p == 5;  2: p == 1 (mod 5)
    u64 d = 0;                  // kind 0: 5^{-1} mod (p-1)
    // kind 2:  p-1 = 5^s * tt,  5 !| tt
    int s5 = 0; u64 tt = 0;
    bool fast2 = false;         // s5 == 1: a residue's root is a^u,  u = 5^{-1} mod tt   (one modpow)
    u64 u = 0;
    // s5 >= 2: Adleman-Manders-Miller style root extraction (see amm_root)
    u64 e1=0, e2=0, alpha=0, gamma=0, gammainv=0;
    u64 zeta_pow[5] = {1,1,1,1,1};
    u64 pow5[16] = {1,5,25,125,625,3125,15625,78125,390625,1953125,9765625,48828125,244140625,1220703125,6103515625ULL,30517578125ULL};
    std::vector<u64> unity;     // the five 5th roots of unity mod p (unsorted powers of zeta)
};
static PrimeInfo make_prime_info(u64 p, std::mt19937_64& /*rng*/){
    PrimeInfo I; I.p = p;
    if (p == 5){ I.kind = 1; return I; }
    if ((p-1) % 5 != 0){ I.kind = 0; I.d = modinv(5, p-1); return I; }
    I.kind = 2;
    // h = first non-fifth-power-residue; zeta = h^((p-1)/5) is a primitive 5th root of unity
    u64 h=2, zeta=1;
    for (;; h++){ zeta = powmod(h, (p-1)/5, p); if (zeta != 1) break; }
    u64 c=1; for (int j=0;j<5;j++){ I.zeta_pow[j]=c; I.unity.push_back(c); c=mulmod(c,zeta,p); }
    u64 tt = p-1; int s=0; while (tt%5==0){ tt/=5; s++; }
    I.s5 = s; I.tt = tt;
    if (s == 1){ I.fast2 = true; I.u = modinv(5 % tt, tt); }
    else {
        u64 pw = I.pow5[s];                                   // 5^s  (s<=11 for p<2^31)
        I.e1 = tt * modinv(tt % pw, pw);                      // == 1 mod 5^s, == 0 mod tt  (< p-1)
        I.e2 = pw * modinv(pw % tt, tt);                      // == 0 mod 5^s, == 1 mod tt  (< p-1)
        I.alpha = modinv(5 % tt, tt);
        I.gamma = powmod(h, tt, p);                           // element of order exactly 5^s
        I.gammainv = powmod(I.gamma, p-2, p);
    }
    return I;
}

// Fifth root of a fifth-power residue a mod p, p == 1 (mod 25).
//   Split a = a5 * at with a5 = a^e1 in the order-5^s subgroup and at = a^e2 in the order-tt subgroup.
//   at has a plain root at^alpha (5 invertible mod tt).  a5 = gamma^(5B): recover B one base-5 digit at a time
//   (Pohlig-Hellman), each digit read off as a power of zeta.  root = at^alpha * gamma^B.
static bool amm_root(u64 a, const PrimeInfo& I, u64* out){
    const u64 p = I.p; const int s = I.s5;
    u64 a5 = powmod(a, I.e1, p), at = powmod(a, I.e2, p);
    u64 xt = powmod(at, I.alpha, p);
    u64 Y = a5, B = 0, p5i = 1;                                // p5i = 5^(i-1)
    for (int i=1;i<s;i++){
        u64 w = powmod(Y, I.pow5[s-1-i], p);
        int dgt=-1; for (int j=0;j<5;j++) if (I.zeta_pow[j]==w){ dgt=j; break; }
        if (dgt < 0) return false;
        B += (u64)dgt * p5i;
        if (dgt) Y = mulmod(Y, powmod(I.gammainv, (u64)dgt * p5i * 5, p), p);
        p5i *= 5;
    }
    *out = mulmod(xt, powmod(I.gamma, B, p), p);
    return true;
}

// All x mod p with x^5 == a (p prime). `info` (optional) supplies the per-prime precomputation.
static std::vector<u64> roots_5th_mod_prime(u64 a,u64 p,std::mt19937_64&rng, const PrimeInfo* info=nullptr){
    a%=p;
    if (a==0) return {0};
    if (p==5) return {a%5};
    PrimeInfo local;
    if (!info){ local = make_prime_info(p, rng); info = &local; }
    if (info->kind == 0) return { powmod(a, info->d, p) };
    // p == 1 (mod 5): a is a fifth power iff a^((p-1)/5) == 1
    if (powmod(a, (p-1)/5, p) != 1) return {};
    u64 r0;
    if (info->fast2){
        // a^t == 1 and gcd(5,t)==1  =>  (a^u)^5 = a^(5u) = a * (a^t)^k = a
        r0 = powmod(a, info->u, p);
    } else if (!amm_root(a, *info, &r0)){
        return {};                                            // unreachable for a genuine 5th-power residue
    }
    std::vector<u64> out; out.reserve(5);
    for (u64 z : info->unity) out.push_back(mulmod(r0,z,p));
    std::sort(out.begin(),out.end()); out.erase(std::unique(out.begin(),out.end()),out.end());
    return out;
}
static std::vector<u64> roots_5th_mod_prime_power(u64 a,u64 p,int k,std::mt19937_64&rng, const PrimeInfo* info=nullptr){
    u64 modk=p; for(int i=1;i<k;i++) modk*=p;
    u64 a_top = a % modk;
    std::vector<u64> roots = roots_5th_mod_prime(a%p, p, rng, info);
    if (roots.empty()) return {};
    u64 cur_mod = p;
    if (p==5){
        for (int j=2;j<=k;j++){
            u64 m=cur_mod*5, aj=a_top%m;
            std::vector<u64> next;
            for (u64 r: roots) for (u64 t=0;t<5;t++){ u64 cand=r+cur_mod*t; if (powmod(cand,5,m)==aj) next.push_back(cand); }
            roots=next; cur_mod=m; if (roots.empty()) return {};
        }
    } else {
        for (int j=1;j<k;j++){
            u64 m_next=cur_mod*p, aj=a_top%m_next;
            std::vector<u64> next;
            for (u64 r: roots){
                if (r%p != 0){
                    u64 r5=powmod(r,5,m_next);
                    i128 diff=((i128)r5-(i128)aj)%(i128)m_next; if (diff<0) diff+=m_next;
                    u64 cdiff=(u64)(diff/cur_mod)%p;
                    u64 r4=powmod(r,4,p);
                    u64 fprime=mulmod(5%p,r4,p);
                    u64 finv=modinv(fprime,p);
                    u64 t=mulmod((p-cdiff%p)%p, finv, p);
                    next.push_back(r+cur_mod*t);
                } else {
                    for (u64 t=0;t<p;t++){ u64 cand=r+cur_mod*t; if (powmod(cand,5,m_next)==aj) next.push_back(cand); }
                }
            }
            roots=next; cur_mod=m_next; if (roots.empty()) return {};
        }
    }
    std::sort(roots.begin(),roots.end()); roots.erase(std::unique(roots.begin(),roots.end()),roots.end());
    return roots;
}

// factor S via SPF sieve
// Smallest-prime-factor sieve over [1, LIMIT]. Linear (O(LIMIT)) time and space.
// Needed so factoring each S in the main loop is O(log S) instead of O(sqrt(S));
// at RANGE=10^8 the latter (trial division to ~14142) would cost ~10^11 ops.
struct SpfSieve {
    std::vector<u32> spf; // spf[i] = smallest prime factor of i, for i in [0,limit]
    void build(u64 limit){
        spf.assign(limit+1, 0);
        std::vector<u32> primes;
        primes.reserve((size_t)(limit/10 + 100)); // rough overestimate of pi(limit); just a perf hint
        for (u64 i=2;i<=limit;i++){
            if (spf[i]==0){ spf[i]=(u32)i; primes.push_back((u32)i); }
            for (u32 p : primes){
                if ((u64)p > spf[i] || i*(u64)p > limit) break;
                spf[i*p] = p;
            }
        }
    }
};

static void factorize_with_spf(u64 S, const SpfSieve& sieve, std::vector<std::pair<u64,int>>& out){
    out.clear();
    while (S > 1){
        u64 p = sieve.spf[S];
        int k = 0;
        while (S % p == 0){ S /= p; k++; }
        out.push_back({p,k});
    }
}
// CRT combine
static inline void crt_combine(u64 r1,u64 m1,u64 r2,u64 m2, u64& r_out, u64& m_out){
    // m1,m2 coprime.  x = r1 (mod m1), x = r2 (mod m2).
    // x = r1 + m1 * t,  t = (r2-r1) * inverse(m1 mod m2)  (mod m2)
    u64 inv = modinv(m1 % m2, m2);
    i128 diff = (i128)r2 - (i128)r1;
    i128 diff_mod = diff % (i128)m2;
    if (diff_mod < 0) diff_mod += m2;
    u128 t = ((u128)diff_mod * inv) % m2;
    u128 x = (u128)r1 + (u128)m1 * t;
    m_out = m1*m2;
    r_out = (u64)(x % m_out);
}

// all x mod S with x^5 == a (mod S), given S's prime-power factorization and
// the matching per-prime precomputation (PrimeInfo)
static std::vector<u64> roots_mod_S(u64 a, u64 S, const std::vector<std::pair<u64,int>>& fac,
                                     const std::vector<PrimeInfo>& infos, std::mt19937_64& rng){
    if (S==1) return {0};
    std::vector<std::vector<u64>> per_factor_roots; // roots mod each p^k
    std::vector<u64> moduli;
    for (size_t fi=0; fi<fac.size(); fi++){
        auto [p,k] = fac[fi];
        u64 pk=p; for(int i=1;i<k;i++) pk*=p;
        auto rts = roots_5th_mod_prime_power(a % pk, p, k, rng, &infos[fi]);
        if (rts.empty()) return {};
        per_factor_roots.push_back(rts);
        moduli.push_back(pk);
    }
    std::vector<u64> combo_r = {0}, combo_m = {1};
    for (size_t i=0;i<per_factor_roots.size();i++){
        std::vector<u64> new_r, new_m;
        for (size_t j=0;j<combo_r.size();j++){
            for (u64 root : per_factor_roots[i]){
                u64 r_out,m_out;
                crt_combine(combo_r[j], combo_m[j], root, moduli[i], r_out, m_out);
                new_r.push_back(r_out); new_m.push_back(m_out);
            }
        }
        combo_r = new_r; combo_m = new_m;
    }
    std::sort(combo_r.begin(), combo_r.end());
    combo_r.erase(std::unique(combo_r.begin(),combo_r.end()), combo_r.end());
    return combo_r;
}

// ------- cheap quadratic-residue pre-filter (avoids the 256-bit pipeline for
// the ~99% of candidates that can be rejected from a single small-modulus check) -----
// QR_MOD is highly composite so squares are unusually rare among its residues
// (~1.1% here, vs. ~50% for a random modulus), which is what makes the filter
// worth its own cost. Modular reduction is a ring homomorphism, so "Disc mod M"
// is computed correctly by this cheap arithmetic regardless of Disc's true sign;
// negative Disc is still caught (a little later, for free) by the sign check in
// solve_candidate -- this filter never needs to know the sign to be correct.
static const u64 QR_MOD = 720720; // = 2^4 * 3^2 * 5 * 7 * 11 * 13
static std::vector<char> g_qr_bitset;
static void build_qr_bitset(){
    g_qr_bitset.assign(QR_MOD, 0);
    for (u64 x=0;x<QR_MOD;x++) g_qr_bitset[(x*x)%QR_MOD] = 1;
}
static inline bool disc_could_be_square(i64 s, i64 z, int n){
    const i64 M = (i64)QR_MOD;
    i64 sp = s % M; if (sp<0) sp+=M;
    i64 zp = z % M; if (zp<0) zp+=M;
    i64 npv = n % M;
    auto mulm=[&](i64 a,i64 b)->i64{ return (a*b) % M; };   // operands < M < 2^20: no overflow
    auto pow5=[&](i64 base)->i64{ i64 b2=mulm(base,base); i64 b4=mulm(b2,b2); return mulm(b4,base); };
    i64 z5p = pow5(zp);
    i64 mp = ((npv - z5p) % M + M) % M;
    i64 fourmp = mulm(4,mp);
    i64 s5p = pow5(sp);
    i64 sump = (s5p+fourmp) % M;
    i64 fivesp = mulm(5,sp);
    i64 discp = mulm(fivesp, sump);
    return g_qr_bitset[discp] != 0;
}

// candidate (s,z) solver
struct Solution { i64 x,y,z; int n; };

static inline bool is_perfect_5th_power(i64 n, i64* c){
    for (i64 v=0; v*v*v*v*v <= n; v++) if (v*v*v*v*v == n){ *c=v; return true; }
    return false;
}
// true if (x,y,z) is a member of the trivial (a,-a,c) family for this n
// (checked in any of the 3 slot assignments)
static inline bool is_trivial_family_member(i64 x, i64 y, i64 z, int n){
    i64 c;
    i64 vals[3] = {x,y,z};
    for (int skip=0; skip<3; skip++){
        i64 u = vals[(skip+1)%3], v = vals[(skip+2)%3], w = vals[skip];
        if (u == -v && is_perfect_5th_power(n,&c) && w==c) return true;
    }
    return false;
}

static int solve_candidate(i64 s, i64 z, int n, i64 RANGE, Solution* out /*size>=2*/){
    if (!disc_could_be_square(s,z,n)) return 0;
    Z256 z5 = pow5_Z256(z);
    Z256 nZ = z256_from_i64(n);
    Z256 m = z256_sub(nZ, z5);
    Z256 fourm = z256_add(z256_add(m,m), z256_add(m,m));
    Z256 s5 = pow5_Z256(s);
    Z256 s5plus4m = z256_add(s5, fourm);
    Z256 Disc = z256_mul_small(s5plus4m, 5*s);
    if (Disc.neg) return 0;
    bool exact;
    u128 D = isqrt_u256(Disc.mag, &exact);
    if (!exact) return 0;
    i128 s3 = (i128)s*s*s;
    i128 fives3 = 5*s3;
    i128 tens = (i128)10*s;
    int cnt=0;
    for (int sign : {1,-1}){
        i128 Dsig = sign>0? (i128)D : -(i128)D;
        i128 numer = fives3 + Dsig;
        if (numer % tens != 0) continue;
        i128 p = numer/tens;
        i128 disc2 = (i128)s*s - 4*p;
        if (disc2<0) continue;
        u128 uv=(u128)disc2;
        long double rd = sqrtl((long double)uv);
        u128 rr = (rd < 0? 0 : (u128)rd);
        while (rr>0 && rr*rr>uv) rr--;
        while ((rr+1)*(rr+1)<=uv) rr++;
        i128 e=(i128)rr;
        if (e*e != disc2) continue;
        if (((i128)s+e)%2 != 0) continue;
        i128 x=((i128)s+e)/2, y=((i128)s-e)/2;
        if (x<-(i128)RANGE || x>(i128)RANGE) continue;
        if (y<-(i128)RANGE || y>(i128)RANGE) continue;
        if (cnt<2){ out[cnt].x=(i64)x; out[cnt].y=(i64)y; out[cnt].z=z; out[cnt].n=n; cnt++; }
    }
    return cnt;
}

// main search
struct SearchResult {
    std::vector<Solution> sporadic;   // deduplicated, trivial-family members excluded
    std::vector<int> trivial_family_ns;
    long long candidates = 0;         // (s,z) candidates examined
};

static void dedup_key(const Solution& s, i64 key[4]){
    i64 v[3]={s.x,s.y,s.z}; std::sort(v,v+3);
    key[0]=v[0]; key[1]=v[1]; key[2]=v[2]; key[3]=s.n;
}

// Largest a >= 0 with a^5 <= n.
static i64 fifth_root_floor(i64 n){ i64 a=0; while ((a+1)*(a+1)*(a+1)*(a+1)*(a+1) <= n) a++; return a; }

// Solutions where some variable is tiny (|v| <= A0, A0^5 <= n_max < (A0+1)^5).
//
// Then the other two satisfy y^5+z^5 = m' = n - v^5 with |m'| <= 2*n_max.
//   * m' == 0            -> the trivial family (n = v^5, y = -z): not enumerated.
//   * m' != 0            -> |y^5+z^5| = |y+z| * Q >= Q,  Q = y^4-y^3z+y^2z^2-yz^3+z^4 = a^2 - ab - b^2
//                           with a = y^2+z^2, b = yz, |b| <= a/2, so Q >= a^2/4 >= max(|y|,|z|)^4 / 4.
//                           Hence max(|y|,|z|) <= (8 n_max)^(1/4): a tiny box.
// So every non-family solution with a tiny variable lives in a small box that we
// simply enumerate.  Everything else is "generic" and handled by the pruned search.
static void small_solutions(i64 RANGE, const std::vector<int>& n_values, std::vector<Solution>& out){
    int n_max = 0; for (int n : n_values) n_max = std::max(n_max, n);
    std::vector<char> want(n_max+1, 0); for (int n : n_values) want[n] = 1;
    i64 A0 = std::min<i64>(fifth_root_floor(n_max), RANGE);
    i64 B  = std::min<i64>((i64)std::floor(std::pow(8.0*n_max, 0.25)) + 2, RANGE);
    auto p5 = [](i64 v)->i128{ i128 a=v; return a*a*a*a*a; };
    for (i64 v=-A0; v<=A0; v++)
        for (i64 y=-B; y<=B; y++)
            for (i64 z=y; z<=B; z++){
                i128 t = p5(v)+p5(y)+p5(z);
                if (t<1 || t>n_max || !want[(int)t]) continue;
                if (is_trivial_family_member(v,y,z,(int)t)) continue;
                out.push_back({v,y,z,(int)t});
            }
}

// Generic search, with the sign/size structure of a solution exploited.
//
// Let |a|<=|b|<=|c| all exceed A0 (so each fifth power exceeds n_max).  Then
// a,b share a sign and c has the opposite sign (any other sign pattern makes some
// fifth power exceed |c|^5).  Call c = z (the largest) and x,y = a,b:
//
//   A:  z = -Z < 0,  x,y > 0 :  x^5 + y^5 = Z^5 + n,    s = x+y = +S
//   B:  z = +Z > 0,  x,y < 0 :  X^5 + Y^5 = Z^5 - n,    s = x+y = -S
//
// Power-mean:  S^5/16 <= x^5+y^5 <= S^5 gives  Z in [ 2^(-4/5) S - 1 , S ]  -- a
// window narrower than one period of the modulus S!  So for each (S, n, root r
// of Z^5 == +-n mod S) there is at most ONE candidate Z instead of ~RANGE/S, and
// S only needs to reach 2^(4/5)*RANGE instead of 2*RANGE.  Candidate count drops
// from O(R log R) to O(R): about 1.2e10 instead of 4e11 for all 99 n at R=10^8.
// S_begin..S_end restricts the sweep over s=|x+y| to a slice (default: everything), so a long run can be cut into
// independent, resumable pieces; results of slices are simply unioned.
static SearchResult run_search(i64 RANGE, const std::vector<int>& n_values, int verbosity, u64 sieve_limit_override=0,
                               long long S_begin=1, long long S_end=-1){
    SearchResult result;
    for (int n : n_values){ i64 c; if (is_perfect_5th_power(n,&c)) result.trivial_family_ns.push_back(n); }
    int n_max = 0; for (int n : n_values) n_max = std::max(n_max, n);
    const i64 G = fifth_root_floor(n_max) + 1;                 // generic solutions have every |var| >= G
    const long double C_HI = 1.7411011265922482L;              // 2^(4/5)
    const long double C_LO = 0.5743491774985174L;              // 2^(-4/5)

    std::vector<Solution> collected;
    small_solutions(RANGE, n_values, collected);

    u64 S_MAX = (u64)(C_HI * (long double)RANGE) + 3;
    if (S_end < 0 || S_end > (long long)S_MAX) S_end = (long long)S_MAX;
    if (S_begin < 1) S_begin = 1;
    const long long S_TOTAL = std::max<long long>(1, S_end - S_begin + 1);
    if (verbosity) printf("Building smallest-prime-factor sieve up to %lld ...\n", S_end);
    SpfSieve sieve;
    auto t0 = std::chrono::steady_clock::now();
    sieve.build(sieve_limit_override? sieve_limit_override : (u64)S_end);
    auto t1 = std::chrono::steady_clock::now();
    if (verbosity) printf("Sieve built in %.2fs\n", std::chrono::duration<double>(t1-t0).count());

    int max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    std::vector<std::vector<Solution>> thread_local_sols(max_threads);

    std::atomic<long long> candidates_examined{0};
    std::atomic<long long> S_done{0};
    auto t_start = std::chrono::steady_clock::now();
    double last_report = 0.0;

    #pragma omp parallel for schedule(dynamic, 256)
    for (long long S = S_begin; S <= S_end; S++){
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const i64 Zlo = std::max<i64>(G, (i64)floorl(C_LO*(long double)S) - 1);
        const i64 ZhiB = std::min<i64>(S,   RANGE);
        const i64 ZhiA = std::min<i64>(S-1, RANGE);
        long long local_cands = 0;
        if (Zlo <= ZhiB){
            std::mt19937_64 rng((u64)S * 0x9E3779B97F4A7C15ULL ^ 0xD1B54A32D192ED03ULL);
            std::vector<std::pair<u64,int>> fac;
            factorize_with_spf((u64)S, sieve, fac);
            // 5th-roots-of-unity per prime factor: independent of n, so once per S.
            std::vector<PrimeInfo> infos; infos.reserve(fac.size());
            for (auto& f : fac) infos.push_back(make_prime_info(f.first, rng));
            auto try_candidate = [&](i64 s, i64 z, int n){
                local_cands++;
                Solution cand[2];
                int cnt = solve_candidate(s, z, n, RANGE, cand);
                for (int i=0;i<cnt;i++)
                    if (!is_trivial_family_member(cand[i].x, cand[i].y, cand[i].z, n))
                        thread_local_sols[tid].push_back(cand[i]);
            };
            for (int n : n_values){
                auto roots = roots_mod_S(((u64)n) % (u64)S, (u64)S, fac, infos, rng);
                for (u64 r : roots){
                    // Case B: Z == r (mod S), z=+Z, s=-S
                    i64 Z = Zlo + ((((i64)r - Zlo) % S) + S) % S;
                    for (; Z <= ZhiB; Z += S) try_candidate(-(i64)S, Z, n);
                    // Case A: Z == -r (mod S), z=-Z, s=+S
                    i64 rA = (S - (i64)r) % S;
                    Z = Zlo + (((rA - Zlo) % S) + S) % S;
                    for (; Z <= ZhiA; Z += S) try_candidate((i64)S, -Z, n);
                }
            }
        }
        candidates_examined.fetch_add(local_cands, std::memory_order_relaxed);
        long long done_now = S_done.fetch_add(1, std::memory_order_relaxed) + 1;
        if (verbosity && (done_now % 2000 == 0)){
            #pragma omp critical
            {
                auto now = std::chrono::steady_clock::now();
                double el = std::chrono::duration<double>(now-t_start).count();
                if (el - last_report >= 2.0){
                    last_report = el;
                    double frac = (double)done_now / (double)S_TOTAL;
                    printf("  %lld/%lld S-values (%.3f%%)  %.1fs elapsed, %.2fM candidates, ETA ~%.1fs\n",
                           done_now, S_TOTAL, 100.0*frac, el, candidates_examined.load()/1e6,
                           frac>0? el*(1.0/frac - 1.0) : 0.0);
                    fflush(stdout);
                }
            }
        }
    }

    // merge + dedup (unordered triples, per n)
    for (auto& v : thread_local_sols) for (auto& s : v) collected.push_back(s);
    std::sort(collected.begin(), collected.end(), [](const Solution&a, const Solution&b){
        i64 ka[4], kb[4]; dedup_key(a,ka); dedup_key(b,kb);
        for (int i=0;i<4;i++){ if(ka[i]!=kb[i]) return ka[i]<kb[i]; }
        return false;
    });
    collected.erase(std::unique(collected.begin(), collected.end(), [](const Solution&a,const Solution&b){
        i64 ka[4],kb[4]; dedup_key(a,ka); dedup_key(b,kb);
        for(int i=0;i<4;i++){ if (ka[i]!=kb[i]) return false; }
        return true;
    }), collected.end());
    result.sporadic = collected;
    result.candidates = candidates_examined.load();
    if (verbosity) printf("Total (s,z) candidates examined: %lld\n", result.candidates);
    return result;
}

static void print_solution_set(const char* label, const std::vector<Solution>& v){
    printf("%s: %zu solutions\n", label, v.size());
    for (size_t i=0;i<v.size() && i<20; i++) printf("   n=%-3d  (%lld, %lld, %lld)\n", v[i].n,(long long)v[i].x,(long long)v[i].y,(long long)v[i].z);
    if (v.size()>20) printf("   ... (%zu more)\n", v.size()-20);
}

// Fast ground truth for stress tests: O(R^2) over pairs, z recovered by a floating-point fifth root
// and verified exactly.  Returns every triple (sorted) with 1 <= n <= NMAX.
static std::vector<Solution> brute_force_fast(i64 R, int NMAX){
    std::vector<Solution> out;
    auto p5 = [](i64 v)->i128{ i128 a=v; return a*a*a*a*a; };
    const i64 W = fifth_root_floor(NMAX) + 2;
    for (i64 x=-R; x<=R; x++){
        i128 px = p5(x);
        for (i64 y=x; y<=R; y++){
            i128 base = px + p5(y);
            long double t = -(long double)base;
            long double zr = (t>=0)? powl(t,0.2L) : -powl(-t,0.2L);
            i64 z0 = (i64)llroundl(zr);
            i64 w = (std::llabs(z0) > W+2) ? 1 : W+2;
            for (i64 z=z0-w; z<=z0+w; z++){
                if (z<y || z>R) continue;              // canonical x<=y<=z
                i128 n = base + p5(z);
                if (n>=1 && n<=NMAX) out.push_back({x,y,z,(int)n});
            }
        }
    }
    return out;
}

// Compare the search with an O(R^2) ground truth.
//   NMAX <= 99 : the real problem, every n = 1..99.
//   NMAX  > 99 : stress test.  n up to NMAX has far more non-trivial solutions, most of them "generic", so this
//                exercises the pruned path much harder; only n that have solutions (plus `extra` random n and a few
//                perfect fifth powers, to exercise the family filter) are searched.
static bool run_selftest(i64 R, int NMAX, int extra){
    printf("=== SELF-TEST: RANGE=%lld, n up to %d ===\n", (long long)R, NMAX);
    auto bf = brute_force_fast(R, NMAX);
    std::vector<Solution> bf_sp;
    for (auto& s : bf) if (!is_trivial_family_member(s.x,s.y,s.z,s.n)) bf_sp.push_back(s);
    std::vector<char> in_set(NMAX+1,0); std::vector<int> ns;
    for (auto& s : bf_sp) if (!in_set[s.n]){ in_set[s.n]=1; ns.push_back(s.n); }
    size_t with_sol = ns.size();
    std::mt19937_64 rng(4242);
    for (int i=0;i<extra;i++){ int n = 1 + (int)(rng()%NMAX); if(!in_set[n]){ in_set[n]=1; ns.push_back(n);} }
    for (int c : {1,2,3,4,5,6}){ int n=c*c*c*c*c; if (n<=NMAX && !in_set[n]){ in_set[n]=1; ns.push_back(n);} } // exercise family filter
    std::sort(ns.begin(), ns.end());
    if (NMAX <= 99){ ns.clear(); for (int n=1; n<=NMAX; n++) ns.push_back(n); }
    printf("brute force: %zu non-family solutions over %zu distinct n; testing %zu n-values (incl. %d random extras)\n",
           bf_sp.size(), with_sol, ns.size(), extra);
    size_t big=0; for (auto& s : bf_sp){ i64 m=std::min({std::llabs(s.x),std::llabs(s.y),std::llabs(s.z)}); if (m>=fifth_root_floor(NMAX)+1) big++; }
    printf("   of which 'generic' (every |var| > %lld): %zu\n", (long long)fifth_root_floor(NMAX), big);
    auto sr = run_search(R, ns, 0);
    auto key_less = [](const Solution&a, const Solution&b){ i64 ka[4],kb[4]; dedup_key(a,ka); dedup_key(b,kb);
        for(int i=0;i<4;i++){ if(ka[i]!=kb[i]) return ka[i]<kb[i]; } return false; };
    std::sort(bf_sp.begin(), bf_sp.end(), key_less);
    bool ok = (bf_sp.size()==sr.sporadic.size());
    for (size_t i=0; ok && i<bf_sp.size(); i++){ i64 ka[4],kb[4]; dedup_key(bf_sp[i],ka); dedup_key(sr.sporadic[i],kb);
        for(int j=0;j<4;j++) if(ka[j]!=kb[j]) ok=false; }
    printf("fast algorithm found %zu solutions.\n", sr.sporadic.size());
    if (!ok){
        for (auto& s : bf_sp){ bool f=false; for (auto& t : sr.sporadic){ i64 ka[4],kb[4]; dedup_key(s,ka); dedup_key(t,kb);
            if (ka[0]==kb[0]&&ka[1]==kb[1]&&ka[2]==kb[2]&&ka[3]==kb[3]) f=true; }
            if(!f) printf("  MISSED: n=%d (%lld,%lld,%lld)\n", s.n,(long long)s.x,(long long)s.y,(long long)s.z); }
    }
    printf(ok ? "*** SELF-TEST PASSED ***\n" : "*** SELF-TEST FAILED ***\n");
    return ok;
}


// root-finder tests
static bool run_root_tests(){
    std::mt19937_64 rng(31337);
    long checked=0, bad=0;
    auto brute = [&](u64 a, u64 p, int k){
        u64 m=p; for(int i=1;i<k;i++) m*=p;
        std::vector<u64> r; for (u64 x=0;x<m;x++) if (powmod(x,5,m)==a%m) r.push_back(x);
        return r; };
    // 1. exhaustive over all residues for many small prime powers (covers p==1 mod 5, mod 25, mod 125, p=5, tiny p)
    std::vector<u64> ps = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,131,151,181,251,401,601,751};
    for (u64 p : ps) for (int k=1;k<=5;k++){
        u64 m=p; for(int i=1;i<k;i++) m*=p;
        if (m > 40000) continue;
        PrimeInfo info = make_prime_info(p, rng);
        for (u64 a=0;a<m;a++){
            auto mine = roots_5th_mod_prime_power(a,p,k,rng,&info);
            auto bf = brute(a,p,k);
            checked++; if (mine!=bf){ bad++; if(bad<10) printf("MISMATCH p=%llu k=%d a=%llu\n",(unsigned long long)p,k,(unsigned long long)a); }
        }
    }
    printf("exhaustive prime-power tests: checked=%ld bad=%ld\n", checked, bad);
    // 2. large primes of every kind: every root satisfies r^5 == a, the planted x is among them, non-residues give none
    auto is_prime = [](u64 n){ if(n<2) return false; for(u64 d=2; d*d<=n; d++) if(n%d==0) return false; return true; };
    long big_checked=0, big_bad=0;
    for (int trial=0; trial<6000; trial++){
        u64 p;
        do { p = 20000 + rng()%180000000ULL; } while(!is_prime(p));
        PrimeInfo info = make_prime_info(p, rng);
        u64 x = 1 + rng()%(p-1);
        u64 a = powmod(x,5,p);                       // guaranteed 5th-power residue
        auto rts = roots_5th_mod_prime(a,p,rng,&info);
        bool ok = !rts.empty();
        for (u64 r : rts) if (powmod(r,5,p)!=a) ok=false;
        if (info.kind==0 && rts.size()!=1) ok=false;
        if (info.kind==2 && rts.size()!=5) ok=false;
        bool has_x=false; for (u64 r: rts) if (r==x) has_x=true; if(!has_x) ok=false;
        if (info.kind==2){
            // a non-residue must give no roots
            u64 b = a; for (u64 g=2; ; g++){ b = mulmod(a, g, p); if (powmod(b,(p-1)/5,p)!=1) break; }
            if (!roots_5th_mod_prime(b,p,rng,&info).empty()) ok=false;
        }
        big_checked++; if(!ok){ big_bad++; if(big_bad<10) printf("BIG MISMATCH p=%llu kind=%d fast2=%d\n",(unsigned long long)p,info.kind,(int)info.fast2); }
    }
    printf("large-prime tests: checked=%ld bad=%ld  (includes bijective primes, p==1 mod 5, p==1 mod 25 via AMM)\n", big_checked, big_bad);

    // 2b. primes with high 5-adic valuation of p-1 (s = 2,3,4,5,6): exercises every iteration count of amm_root
    {
        long hv_checked=0, hv_bad=0; int seen_s[12]={0};
        for (u64 s=2; s<=6; s++){
            u64 pw=1; for(u64 i=0;i<s;i++) pw*=5;
            int found=0;
            for (u64 m=1; m<2000000 && found<6; m++){
                if (m%5==0) continue;
                u64 p = pw*m+1;
                if (p > 400000000ULL) break;
                if (!is_prime(p)) continue;
                found++;
                PrimeInfo info = make_prime_info(p, rng);
                if (info.s5 != (int)s){ hv_bad++; continue; }
                seen_s[s]++;
                for (int t=0;t<400;t++){
                    u64 x = 1 + rng()%(p-1), a = powmod(x,5,p);
                    auto rts = roots_5th_mod_prime(a,p,rng,&info);
                    bool ok = (rts.size()==5); bool hx=false;
                    for (u64 r: rts){ if (powmod(r,5,p)!=a) ok=false; if (r==x) hx=true; }
                    if (!hx) ok=false;
                    hv_checked++; if(!ok){ hv_bad++; if(hv_bad<10) printf("HV MISMATCH p=%llu s=%llu\n",(unsigned long long)p,(unsigned long long)s); }
                }
            }
        }
        printf("high-valuation primes (s=2..6 seen: %d %d %d %d %d): checked=%ld bad=%ld\n", seen_s[2],seen_s[3],seen_s[4],seen_s[5],seen_s[6], hv_checked, hv_bad);
        big_bad += hv_bad;
    }
    // 3. prime powers with big p (Hensel lift on top of the fast paths)
    long pp_checked=0, pp_bad=0;
    for (int trial=0; trial<3000; trial++){
        u64 p; do { p = 100 + rng()%12000ULL; } while(!is_prime(p));
        int k = 2 + (int)(rng()%2);
        u64 m=p; for(int i=1;i<k;i++) m*=p;
        if (m > 400000000000ULL) continue;
        PrimeInfo info = make_prime_info(p, rng);
        u64 x = 1 + rng()%(m-1);
        if (x%p==0) continue;
        u64 a = powmod(x,5,m);
        auto rts = roots_5th_mod_prime_power(a,p,k,rng,&info);
        bool ok = !rts.empty(); bool has_x=false;
        for (u64 r: rts){ if (powmod(r,5,m)!=a) ok=false; if (r==x) has_x=true; }
        if (!has_x) ok=false;
        pp_checked++; if(!ok){ pp_bad++; if(pp_bad<10) printf("PP MISMATCH p=%llu k=%d\n",(unsigned long long)p,k); }
    }
    printf("large prime-power tests: checked=%ld bad=%ld\n", pp_checked, pp_bad);
    bool all_ok = (bad==0 && big_bad==0 && pp_bad==0);
    printf(all_ok ? "*** ROOT TESTS PASSED ***\n" : "*** ROOT TESTS FAILED ***\n");
    return all_ok;
}


// arithmetic tests at full magnitude
static bool run_arith_tests(){
    std::mt19937_64 rng(2718281828ULL);
    auto rnd128 = [&]()->u128{ return ((u128)rng()<<64)|rng(); };
    long bad=0, total=0;
    // 1. isqrt vs bit-by-bit reference: perfect squares and near-squares with roots up to 2^127, incl. the ~2^84 range that matters
    for (int t=0;t<400000;t++){
        int bits = 1 + (int)(rng()%127);
        u128 k = rnd128() >> (128-bits);
        U256 n = mul128_to_256(k,k);
        int mode = t%5;
        if (mode==1 && k>0){ n = mul128_to_256(k+1,k+1); n = u256_sub(n, u256_from_u128(1)); }        // (k+1)^2-1
        if (mode==2){ U256 add = u256_from_u128(rng()%(2*(u64)(k>>1)+3)); n = u256_add(n, add); }       // k^2 + small
        if (mode==3 && k>0){ n = u256_sub(n, u256_from_u128(1)); }                                       // k^2-1
        bool e1,e2; u128 r1=isqrt_u256(n,&e1), r2=isqrt_u256_reference(n,&e2);
        total++; if (r1!=r2 || e1!=e2){ bad++; if(bad<8) printf("ISQRT MISMATCH bits=%d mode=%d\n",bits,mode); }
    }
    printf("isqrt vs reference (roots up to 2^127, incl. 2^84 regime): %ld cases, bad=%ld\n", total, bad);
    // 2. planted large-magnitude instances: family (a,-a,c): labeling z=a, {x,y}={-a,c}, s=c-a, n=c^5
    long pb=0, pt=0;
    for (int t=0;t<300000;t++){
        i64 a = (i64)(rng()%200000000ULL) + 1; if (rng()&1) a = -a;   // |a| up to 2e8
        i64 c = (i64)(rng()%6000) + 1;                                 // n=c^5 <= 7.8e18 fits i64
        i64 s = c - a; if (s==0) continue;
        i64 n64 = c*c*c*c*c;
        if (n64 > 9000000000000000000LL) continue;
        // n is an int in solve_candidate's signature: widen the test through a local copy of the pipeline instead
        Z256 z5 = pow5_Z256(a); Z256 nZ = z256_from_i64(n64);
        Z256 m = z256_sub(nZ, z5);
        Z256 fourm = z256_add(z256_add(m,m), z256_add(m,m));
        Z256 s5 = pow5_Z256(s);
        Z256 Disc = z256_mul_small(z256_add(s5, fourm), 5*s);
        pt++;
        bool ok = !Disc.neg;
        bool exact=false; u128 D = 0;
        if (ok){ D = isqrt_u256(Disc.mag, &exact); ok = exact; }
        if (ok){
            // recover x,y exactly as solve_candidate does
            i128 s3=(i128)s*s*s, fives3=5*s3, tens=(i128)10*s; bool found=false;
            for (int sign : {1,-1}){
                i128 Ds = sign>0?(i128)D:-(i128)D; i128 numer=fives3+Ds;
                if (numer%tens) continue;
                i128 p=numer/tens; i128 d2=(i128)s*s-4*p;
                if (d2<0) continue;
                u128 uv=(u128)d2; long double rd=sqrtl((long double)uv); u128 rr=(u128)rd;
                while (rr>0 && rr*rr>uv) rr--;
                while ((rr+1)*(rr+1)<=uv) rr++;
                if ((i128)(rr*rr)!=d2) continue;
                if (((i128)s+(i128)rr)%2) continue;
                i128 x=((i128)s+(i128)rr)/2, y=((i128)s-(i128)rr)/2;
                if ((x==-a && y==c) || (x==c && y==-a)) found=true;
            }
            ok = found;
        }
        if (!ok){ pb++; if (pb<8) printf("PLANTED FAIL a=%lld c=%lld\n",(long long)a,(long long)c); }
    }
    printf("planted large-magnitude family instances (|a| up to 2e8): %ld cases, bad=%ld\n", pt, pb);
    bool all = (bad==0 && pb==0);
    printf(all ? "*** ARITHMETIC TESTS PASSED ***\n" : "*** ARITHMETIC TESTS FAILED ***\n");
    return all;
}

// main
static void usage(const char* prog){
    fprintf(stderr,
      "Usage:\n"
      "  %s --selftest [RANGE] [NMAX] [EXTRA]   cross-check against brute force (defaults 3000 99 0);\n"
      "                                        NMAX > 99 makes it a stress test with n up to NMAX\n"
      "  %s --test                              unit tests: fifth-root finder and 256-bit layer\n"
      "  %s --run RANGE [out.txt] [S_begin S_end]\n"
      "                                        search |x|,|y|,|z| <= RANGE for n = 1..99; optionally only the slice\n"
      "                                        S_begin..S_end of the s = |x+y| sweep (slices are simply unioned)\n",
      prog, prog, prog);
}

int main(int argc, char** argv){
    build_qr_bitset();
    if (argc < 2){ usage(argv[0]); return 1; }
    std::string mode = argv[1];
    if (mode == "--selftest"){
        i64 R = (argc>=3)? atoll(argv[2]) : 3000;
        int NMAX = (argc>=4)? atoi(argv[3]) : 99;
        int extra = (argc>=5)? atoi(argv[4]) : 0;
        return run_selftest(R, NMAX, extra) ? 0 : 1;
    } else if (mode == "--test"){
        bool roots_ok = run_root_tests();
        bool arith_ok = run_arith_tests();
        return (roots_ok && arith_ok) ? 0 : 1;
    } else if (mode == "--run"){
        if (argc < 3){ usage(argv[0]); return 1; }
        i64 RANGE = atoll(argv[2]);
        std::string outpath = (argc>=4)? argv[3] : "solutions.txt";
        std::vector<int> n_values; for(int n=1;n<100;n++) n_values.push_back(n);

        printf("x^5+y^5+z^5=n search,  RANGE=%lld,  n=1..99\n", (long long)RANGE);
        printf("(cost: about 115*RANGE candidate (s,z) checks and 172*RANGE fifth-root computations for n=1..99)\n");
#ifdef _OPENMP
        printf("OpenMP threads available: %d\n", omp_get_max_threads());
#else
        printf("Built WITHOUT OpenMP (single-threaded). Compile with -fopenmp for parallel search.\n");
#endif
        long long sb = (argc>=5)? atoll(argv[4]) : 1;
        long long se = (argc>=6)? atoll(argv[5]) : -1;
        auto sr = run_search(RANGE, n_values, /*verbosity=*/1, 0, sb, se);

        FILE* f = fopen(outpath.c_str(), "w");
        if (!f){ perror("fopen"); return 1; }
        fprintf(f, "# x^5+y^5+z^5=n ,  RANGE=%lld\n", (long long)RANGE);
        fprintf(f, "# Trivial infinite families (a,-a,c) with c^5=n, any integer a in [-RANGE,RANGE], any of 3 slots:\n");
        for (int n : sr.trivial_family_ns){
            i64 c = 0; is_perfect_5th_power(n,&c);
            fprintf(f, "#   n=%d : (a, -a, %lld) for all integer a, |a| <= RANGE\n", n, (long long)c);
        }
        fprintf(f, "# Sporadic solutions (not of the trivial family form), n x y z:\n");
        for (auto& s : sr.sporadic) fprintf(f, "%d %lld %lld %lld\n", s.n, (long long)s.x, (long long)s.y, (long long)s.z);
        fprintf(f, "# candidates %lld\n# complete\n", sr.candidates);
        fclose(f);

        printf("\nTrivial-family n values: ");
        for (int n : sr.trivial_family_ns) printf("%d ", n);
        printf("\nSporadic solutions found: %zu (written to %s)\n", sr.sporadic.size(), outpath.c_str());
        if (!sr.sporadic.empty()) print_solution_set("sporadic solutions", sr.sporadic);
        return 0;
    } else {
        usage(argv[0]); return 1;
    }
}

