// a^5 + b^5 + c^5 = n over |a|,|b|,|c| <= B, written as x^5 + y^5 - z^5 = n with (x,y,z) = (a,b,-c).
//
// For max(|a|,|b|,|c|) >= zmin the two smaller variables share a sign and the largest has the other,
// so (x,y,z) lies within ~|n|/z^4 of the cone x^5 + y^5 = z^5, on the arc u = x/z in [u_lo, 2^(-1/5)]
// (x <= y; z > y and x^5 > 5y^4 - n give u > 1.23 z^(-1/5)).  Cut that arc into short pieces.  Over a
// piece the cone is within a tiny distance of its tangent plane, so every solution of the piece lies in
//     |z| <= B,   |x - u0 z| <= B h,   |u0^4 x + v0^4 y - z| <= T,        v0^5 = 1 - u0^5,
// with T = B v0^4 (max|v''|/2) h^2 + eps.  The width h is chosen so the box holds about E points of the
// lattice x + y - z = n (mod 30); t^5 = t (mod 30) makes that congruence necessary, and it shrinks the
// number of boxes by 30^(1/3).  LLL on that lattice turns the box into a near-cube whose points are
// listed by Fincke-Pohst.  O(1) work per piece, O(B) pieces.
//
// Exactness: x^5 reaches 3e41 > 2^128 at B = 2e8, so F = x^5 + y^5 - z^5 is evaluated modulo 2^128.
// That equals the true value because points of a box have |F| ~ 5 B^4 T ~ 1e27 < 2^127.  Solutions with
// |a|,|b| <= zmin are found by a direct scan instead.
//
// fifth3 [--n N] [--B B] [--zmin Z] [--E e] [--M 30|1] [--chunks K] [--threads t] [--noscan]
// self-tests: --near NMAX (every point with |F| <= NMAX, forces M = 1), --boxcheck NB --ustart u,
// --audit K (dump plane and thickness of K boxes for a high-precision check)

#include <bits/stdc++.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;

typedef long long ll;
typedef long double ld;
typedef __int128 i128;
typedef unsigned __int128 u128;

static const int FRAC = 80;   // fraction bits of the fixed-point tangent plane

struct Config {
    i128 n = 8;
    ll B = 200000000LL;
    ll zmin = 2000;           // below this the equation is scanned directly
    double E = 5.0;           // expected lattice points per box
    int M = 30;               // lattice x + y - z = 0 (mod M); M = 30 uses x^5 = x (mod 30), M = 1 is plain Z^3
    bool near_mode = false;   // report every point with |x^5 + y^5 - z^5| <= nmax, needs M = 1
    i128 nmax = 0;
    double ulo = -1, uhi = -1;
    int chunks = 2048;
    int threads = 0;
    bool scan = true;
    bool verbose = true;
    int audit = 0;            // print the plane and thickness of this many boxes and exit
    int boxcheck = 0;         // self-test on this many consecutive boxes starting at ustart
    double ustart = 0.5;
};
static Config cfg;

static string str128(i128 v) {
    if (v == 0) return "0";
    bool neg = v < 0;
    u128 u = neg ? (u128)0 - (u128)v : (u128)v;
    string s;
    while (u) { s += char('0' + (int)(u % 10)); u /= 10; }
    if (neg) s += '-';
    reverse(s.begin(), s.end());
    return s;
}

static i128 parse128(const string& s) {
    size_t i = 0;
    bool neg = false;
    if (s[0] == '-') { neg = true; i = 1; }
    i128 v = 0;
    if (s.find('e') != string::npos) {
        double d = atof(s.c_str() + i);
        v = (i128)d;
    } else {
        for (; i < s.size(); ++i) v = v * 10 + (s[i] - '0');
    }
    return neg ? -v : v;
}

// x^5 + y^5 - z^5 modulo 2^128; exact whenever the true value is below 2^127 in size
static inline i128 f5(ll x, ll y, ll z) {
    u128 X = (u128)(i128)x, Y = (u128)(i128)y, Z = (u128)(i128)z;
    u128 X2 = X * X, Y2 = Y * Y, Z2 = Z * Z;
    return (i128)(X2 * X2 * X + Y2 * Y2 * Y - Z2 * Z2 * Z);
}

// curve v = (1 - u^5)^(1/5) and its curvature bound |v''| = 4 u^3 / v^9 (increasing in u)
static inline double curve_v(double u) {
    double u2 = u * u;
    return pow(1.0 - u2 * u2 * u, 0.2);
}
static inline double curve_m2(double u) {
    double v = curve_v(u), v3 = v * v * v;
    return 4.0 * u * u * u / (v3 * v3 * v3);
}

struct Patch {
    double u0d, inv1, inv2, inv3, T, bh;
    i128 afx, bfx;
};

struct Setup {
    ld B;
    double Bd, g, eps_n, eps_num;
};

// state carried from one box to the next along the arc
struct Walker {
    ld uL;            // left end of the next box
    double phiL;      // curve value at uL
    double v0p, dphi; // previous centre value and slope, for a first-order guess
    ld u0p;
    bool have;
};

static const double CBRT4 = 1.5874010519681994748;
static const ld TWO80 = 1208925819614629174706176.0L;

// 0 <= x < 2^100
static inline i128 ld_to_i128(ld x) {
    ld hi = x * 5.421010862427522170037264004349708557128906250e-20L;
    unsigned long long h = (unsigned long long)hi;
    ld rem = x - (ld)h * 18446744073709551616.0L;
    unsigned long long l = (unsigned long long)rem;
    return ((i128)h << 64) | l;
}

static inline double i128_to_double(i128 t) {
    long long hi = (long long)(t >> 48);
    unsigned long long lo = (unsigned long long)t & ((1ULL << 48) - 1);
    return (double)hi * 281474976710656.0 + (double)lo;
}

// one box starting at W.uL: fills P and advances W to the next box; returns the half width h
static ld make_patch(Walker& W, const Setup& S, Patch& P) {
    double uLd = (double)W.uL, phiL = W.phiL;
    double phi2 = phiL * phiL, phi5 = phi2 * phi2 * phiL;
    // g / cbrt(m2(uL)) with m2 = 4 u^3 / phi^9; m2 grows by at most exp(w kappa) over a width w
    double G = S.g * phi2 * phiL / (CBRT4 * uLd);
    double kappa = (3.0 / uLd + 9.0 * uLd * uLd * uLd * uLd / phi5) * 1.001;
    double x = (2.0 * G / S.Bd) / 3.0 * kappa * 1.001;
    double bh = G * (1.0 - x) * (1.0 - 1e-9);
    double gb = S.g / bh;
    double m2p = gb * gb * gb * (1.0 + 1e-6);   // >= max |v''| on the box
    ld h = (ld)bh / S.B;
    ld u0 = W.uL + h;
    ld u2 = u0 * u0, u4 = u2 * u2;
    ld c = 1.0L - u4 * u0;
    ld v = W.have ? (ld)(W.v0p + W.dphi * (double)(u0 - W.u0p)) : (ld)pow((double)c, 0.2);
    {
        ld w2 = v * v, w4 = w2 * w2;
        v -= (w4 * v - c) / (5.0L * w4);
    }
    ld v2 = v * v, v4 = v2 * v2;
    P.u0d = (double)u0;
    P.afx = ld_to_i128(u4 * TWO80);
    P.bfx = ld_to_i128(v4 * TWO80);
    P.bh = bh;
    P.T = (1.0 + 1e-6) * (double)v4 * (0.5 * m2p) * bh * bh / S.Bd + S.eps_n + S.eps_num;
    P.inv1 = 1.0 / S.Bd;
    P.inv2 = 1.0 / (bh * (1.0 + 1e-6));
    P.inv3 = 1.0 / (P.T * ldexp(1.0, FRAC));
    double vd = (double)v, ud = P.u0d, hd = (double)h;
    double dphi = -(double)(u4 / v4);
    double vd3 = vd * vd * vd;
    double d2 = -4.0 * ud * ud * ud / (vd3 * vd3 * vd3);
    W.phiL = vd + dphi * hd + 0.5 * d2 * hd * hd;
    W.uL += 2 * h;
    W.v0p = vd; W.dphi = dphi; W.u0p = u0; W.have = true;
    return h;
}

// scaled coordinates in which the box is the cube [-1,1]^3
static inline void fill_c(const Patch& P, const ll* v, double* c) {
    c[0] = (double)v[2] * P.inv1;
    c[1] = ((double)v[0] - P.u0d * (double)v[2]) * P.inv2;
    i128 t = P.afx * v[0] + P.bfx * v[1] - ((i128)v[2] << FRAC);
    c[2] = i128_to_double(t) * P.inv3;
}

struct Lat {
    ll b[3][3];
    double c[3][3];
    ll r[3];
    double cr[3];
    double bstar[3][3];
    double bn[3];
    double mu[3][3];
};

static inline double dot3(const double* a, const double* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline void gs_row(Lat& L, int i) {
    double v[3] = {L.c[i][0], L.c[i][1], L.c[i][2]};
    for (int j = 0; j < i; ++j) {
        double m = dot3(v, L.bstar[j]) / L.bn[j];
        L.mu[i][j] = m;
        v[0] -= m * L.bstar[j][0];
        v[1] -= m * L.bstar[j][1];
        v[2] -= m * L.bstar[j][2];
    }
    L.bstar[i][0] = v[0]; L.bstar[i][1] = v[1]; L.bstar[i][2] = v[2];
    L.bn[i] = dot3(v, v);
}

// LLL (delta = 0.99) for the current box, started from the previous box's basis
static bool lll3(Lat& L, const Patch& P, ll& steps) {
    for (int i = 0; i < 3; ++i) fill_c(P, L.b[i], L.c[i]);
    gs_row(L, 0);
    int k = 1, guard = 0;
    while (k < 3) {
        if (++guard > 4000) return false;
        for (int i = 1; i <= k; ++i) gs_row(L, i);
        for (int j = k - 1; j >= 0; --j) {
            double m = L.mu[k][j];
            if (fabs(m) > 0.5) {
                if (fabs(m) > 1e18) return false;
                ll q = llround(m);
                for (int t = 0; t < 3; ++t) L.b[k][t] -= q * L.b[j][t];
                fill_c(P, L.b[k], L.c[k]);
                gs_row(L, k);
            }
        }
        if (L.bn[k] >= (0.99 - L.mu[k][k - 1] * L.mu[k][k - 1]) * L.bn[k - 1]) {
            ++k;
        } else {
            for (int t = 0; t < 3; ++t) {
                swap(L.b[k][t], L.b[k - 1][t]);
                swap(L.c[k][t], L.c[k - 1][t]);
            }
            k = max(k - 1, 1);
            gs_row(L, 0);
        }
    }
    steps += guard;
    return true;
}

struct Hit { ll x, y, z; i128 F; };

struct Stats {
    ll patches = 0, steps = 0, leaves = 0, boxed = 0, exact = 0, fail = 0;
    void add(const Stats& o) {
        patches += o.patches; steps += o.steps; leaves += o.leaves;
        boxed += o.boxed; exact += o.exact; fail += o.fail;
    }
};

// fifth powers mod 11 and 31 as a cheap prefilter for the exact value n
struct Sieve {
    int pw11[11], pw31[31];
    int t11, t31;
    void init(i128 n) {
        for (int i = 0; i < 11; ++i) pw11[i] = (int)(((ll)i * i % 11) * ((ll)i * i % 11) % 11 * i % 11);
        for (int i = 0; i < 31; ++i) pw31[i] = (int)(((ll)i * i % 31) * ((ll)i * i % 31) % 31 * i % 31);
        t11 = (int)(((n % 11) + 11) % 11);
        t31 = (int)(((n % 31) + 31) % 31);
    }
    inline bool ok(ll x, ll y, ll z) const {
        int a = pw11[(int)((x % 11 + 11) % 11)], b = pw11[(int)((y % 11 + 11) % 11)],
            c = pw11[(int)((z % 11 + 11) % 11)];
        if ((a + b + 11 - c) % 11 != t11) return false;
        a = pw31[(int)((x % 31 + 31) % 31)];
        b = pw31[(int)((y % 31 + 31) % 31)];
        c = pw31[(int)((z % 31 + 31) % 31)];
        return (a + b + 31 - c) % 31 == t31;
    }
};

static void init_lat(Lat& L) {
    memset(&L, 0, sizeof(L));
    L.b[0][0] = 1; L.b[0][2] = 1;
    L.b[1][1] = 1; L.b[1][2] = 1;
    L.b[2][2] = cfg.M;
    L.r[0] = (ll)(((cfg.n % cfg.M) + cfg.M) % cfg.M);
}

// reduce, bring the coset representative close to the box centre, then list the lattice points of the
// coset inside the cube by Fincke-Pohst; emit(x, y, z) is called for each one
template <class Emit>
static bool process_box(Lat& L, const Patch& P, Stats& st, Emit&& emit) {
    const double Rb = 1.0 + 1e-5;
    const double R2 = 3.0 * Rb * Rb;
    if (!lll3(L, P, st.steps)) return false;
    fill_c(P, L.r, L.cr);
    for (int i = 2; i >= 0; --i) {
        double m = dot3(L.cr, L.bstar[i]) / L.bn[i];
        if (fabs(m) > 0.5) {
            if (fabs(m) > 1e18) return false;
            ll q = llround(m);
            for (int t = 0; t < 3; ++t) L.r[t] -= q * L.b[i][t];
            fill_c(P, L.r, L.cr);
        }
    }
    double tau0 = -dot3(L.cr, L.bstar[0]) / L.bn[0];
    double tau1 = -dot3(L.cr, L.bstar[1]) / L.bn[1];
    double tau2 = -dot3(L.cr, L.bstar[2]) / L.bn[2];
    double s2 = tau2;
    double s1 = tau1 - L.mu[2][1] * s2;
    double s0 = tau0 - L.mu[1][0] * s1 - L.mu[2][0] * s2;
    double ext2 = sqrt(R2 / L.bn[2]);
    double lo2 = ceil(s2 - ext2), hi2 = floor(s2 + ext2);
    if (hi2 - lo2 > 1e6) return false;
    for (ll k2 = (ll)lo2; k2 <= (ll)hi2; ++k2) {
        double y2 = k2 - s2;
        double d2 = L.bn[2] * y2 * y2;
        if (d2 > R2) continue;
        double cen1 = s1 - L.mu[2][1] * y2;
        double ext1 = sqrt((R2 - d2) / L.bn[1]);
        double lo1 = ceil(cen1 - ext1), hi1 = floor(cen1 + ext1);
        if (hi1 - lo1 > 1e6) return false;
        for (ll k1 = (ll)lo1; k1 <= (ll)hi1; ++k1) {
            double e1 = k1 - cen1;
            double d1 = d2 + L.bn[1] * e1 * e1;
            if (d1 > R2) continue;
            double y1 = k1 - s1;
            double cen0 = s0 - L.mu[1][0] * y1 - L.mu[2][0] * y2;
            double ext0 = sqrt((R2 - d1) / L.bn[0]);
            double lo0 = ceil(cen0 - ext0), hi0 = floor(cen0 + ext0);
            if (hi0 - lo0 > 1e6) return false;
            for (ll k0 = (ll)lo0; k0 <= (ll)hi0; ++k0) {
                ++st.leaves;
                double a0 = L.cr[0] + k0 * L.c[0][0] + k1 * L.c[1][0] + k2 * L.c[2][0];
                double a1 = L.cr[1] + k0 * L.c[0][1] + k1 * L.c[1][1] + k2 * L.c[2][1];
                double a2 = L.cr[2] + k0 * L.c[0][2] + k1 * L.c[1][2] + k2 * L.c[2][2];
                if (fabs(a0) > Rb || fabs(a1) > Rb || fabs(a2) > Rb) continue;
                ++st.boxed;
                emit(L.r[0] + k0 * L.b[0][0] + k1 * L.b[1][0] + k2 * L.b[2][0],
                     L.r[1] + k0 * L.b[0][1] + k1 * L.b[1][1] + k2 * L.b[2][1],
                     L.r[2] + k0 * L.b[0][2] + k1 * L.b[1][2] + k2 * L.b[2][2]);
            }
        }
    }
    return true;
}

static void run_chunk(ld uStart, ld uEnd, const Setup& S, const Sieve& sv, Stats& st, vector<Hit>& hits) {
    Lat L;
    init_lat(L);
    Walker W;
    W.uL = uStart;
    W.phiL = curve_v((double)uStart);
    W.have = false;
    auto emit = [&](ll x, ll y, ll z) {
        if (llabs(z) < cfg.zmin || llabs(z) > cfg.B || llabs(x) > cfg.B || llabs(y) > cfg.B) return;
        if (cfg.near_mode) {
            ++st.exact;
            i128 F = f5(x, y, z);
            if (F >= -cfg.nmax && F <= cfg.nmax) hits.push_back({x, y, z, F});
        } else {
            if (!sv.ok(x, y, z)) return;
            ++st.exact;
            i128 F = f5(x, y, z);
            if (F == cfg.n) hits.push_back({x, y, z, F});
        }
    };
    while (W.uL < uEnd) {
        Patch P;
        make_patch(W, S, P);
        ++st.patches;
        if (!process_box(L, P, st, emit)) { ++st.fail; return; }
    }
}

// self-test: nb consecutive boxes from ustart, Fincke-Pohst against an exhaustive scan over (z, x)
static int box_check(const Setup& S, double ustart, int nb) {
    Lat L;
    init_lat(L);
    Walker W;
    W.uL = ustart;
    W.phiL = curve_v(ustart);
    W.have = false;
    Stats st;
    const double inner = 1.0 + 2e-6, outer = 1.0 + 2e-5;
    ll pts = 0, bad = 0;
    for (int bi = 0; bi < nb; ++bi) {
        Patch P;
        make_patch(W, S, P);
        vector<array<ll, 3>> got;
        if (!process_box(L, P, st, [&](ll x, ll y, ll z) { got.push_back({x, y, z}); })) {
            printf("reduction failed in box %d\n", bi);
            return 2;
        }
        ld alpha = (ld)P.afx / TWO80, beta = (ld)P.bfx / TWO80;
        ll Z = (ll)(cfg.B * outer) + 1;
        double Rx = P.bh * (1 + 1e-6) * outer + 1e-3;
        const int M = cfg.M;
        ll r = ((cfg.n % M) + M) % M;
        set<array<ll, 3>> in_set, out_set;
        for (ll z = -Z; z <= Z; ++z) {
            ll xlo = (ll)ceil(P.u0d * z - Rx), xhi = (ll)floor(P.u0d * z + Rx);
            for (ll x = xlo; x <= xhi; ++x) {
                ll y = llroundl(((ld)z - alpha * x) / beta);
                if ((((x + y - z) % M) + M) % M != r) continue;
                ll v[3] = {x, y, z};
                double c[3];
                fill_c(P, v, c);
                double m = max({fabs(c[0]), fabs(c[1]), fabs(c[2])});
                if (m <= outer) out_set.insert({x, y, z});
                if (m <= inner) in_set.insert({x, y, z});
            }
        }
        set<array<ll, 3>> gs(got.begin(), got.end());
        for (auto& p : in_set) if (!gs.count(p)) { ++bad; printf("box %d: missed (%lld,%lld,%lld)\n", bi, p[0], p[1], p[2]); }
        for (auto& p : gs) if (!out_set.count(p)) { ++bad; printf("box %d: spurious (%lld,%lld,%lld)\n", bi, p[0], p[1], p[2]); }
        pts += (ll)in_set.size();
    }
    printf("boxes %d from u=%.6f: %lld lattice points compared, %lld discrepancies\n", nb, ustart, pts, bad);
    return bad ? 1 : 0;
}

// direct scan of every (a,b,c) with |a|,|b| <= zs; c is the exact fifth root of n - a^5 - b^5
static void small_scan(ll zs, set<array<ll, 3>>& out) {
    for (ll a = -zs; a <= zs; ++a) {
        i128 a5 = (i128)a * a * a * a * a;
        for (ll b = a; b <= zs; ++b) {
            i128 r = cfg.n - a5 - (i128)b * b * b * b * b;
            if (r == 0) { array<ll, 3> t = {a, b, 0}; sort(t.begin(), t.end()); out.insert(t); continue; }
            bool neg = r < 0;
            u128 m = neg ? (u128)0 - (u128)r : (u128)r;
            ll c0 = (ll)llroundl(powl((ld)m, 0.2L));
            for (ll c = max(1LL, c0 - 1); c <= c0 + 1; ++c) {
                u128 c2 = (u128)c * c, c5 = c2 * c2 * c;
                if (c5 == m) {
                    array<ll, 3> t = {a, b, neg ? -c : c};
                    sort(t.begin(), t.end());
                    out.insert(t);
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        auto next = [&]() { return string(argv[++i]); };
        if (a == "--n") cfg.n = parse128(next());
        else if (a == "--B") cfg.B = (ll)parse128(next());
        else if (a == "--zmin") cfg.zmin = (ll)parse128(next());
        else if (a == "--E") cfg.E = atof(next().c_str());
        else if (a == "--M") cfg.M = atoi(next().c_str());
        else if (a == "--near") { cfg.near_mode = true; cfg.nmax = parse128(next()); }
        else if (a == "--ulo") cfg.ulo = atof(next().c_str());
        else if (a == "--uhi") cfg.uhi = atof(next().c_str());
        else if (a == "--chunks") cfg.chunks = atoi(next().c_str());
        else if (a == "--threads") cfg.threads = atoi(next().c_str());
        else if (a == "--noscan") cfg.scan = false;
        else if (a == "--quiet") cfg.verbose = false;
        else if (a == "--audit") cfg.audit = atoi(next().c_str());
        else if (a == "--boxcheck") cfg.boxcheck = atoi(next().c_str());
        else if (a == "--ustart") cfg.ustart = atof(next().c_str());
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }
    if (cfg.near_mode) cfg.M = 1;
    if (cfg.B > 2000000000LL)
        fprintf(stderr, "warning: B > 2e9, the long double tangent plane error (~1e-19 B) approaches the box thickness (~1/B); results stay correct but slow\n");
#ifdef _OPENMP
    if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
#endif
    double t0 = chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();

    Setup S;
    S.B = (ld)cfg.B;
    S.Bd = (double)cfg.B;
    S.g = cbrt(cfg.E * cfg.M / 4.0);
    i128 nref = cfg.near_mode ? cfg.nmax : (cfg.n < 0 ? -cfg.n : cfg.n);
    S.eps_n = 0.4 * (double)nref / pow((double)cfg.zmin, 4.0);
    S.eps_num = 2e-18 * S.Bd;
    Sieve sv;
    sv.init(cfg.n);

    double ulo = cfg.ulo >= 0 ? cfg.ulo : 0.9 * pow(S.Bd, -0.2);
    double uhi = cfg.uhi >= 0 ? cfg.uhi : 0.8706;

    if (cfg.boxcheck > 0) return box_check(S, cfg.ustart, cfg.boxcheck);
    if (cfg.audit > 0) {
        for (int i = 0; i < cfg.audit; ++i) {
            ld uL = (ld)ulo + (ld)(uhi - ulo) * (ld)(i + 0.5) / (ld)cfg.audit;
            Walker W;
            W.uL = uL;
            W.phiL = curve_v((double)uL);
            W.have = false;
            Patch P;
            ld h = make_patch(W, S, P);
            printf("%.25Lg %.25Lg %.17g %.17g %s %s\n", uL, uL + h, P.bh, P.T, str128(P.afx).c_str(), str128(P.bfx).c_str());
        }
        return 0;
    }

    // chunk boundaries with equal expected number of boxes: density B m2^(1/3) / (2 g)
    const int NT = 200000;
    vector<double> cum(NT + 1, 0.0);
    for (int i = 1; i <= NT; ++i) {
        double u1 = ulo + (uhi - ulo) * (i - 1) / NT, u2 = ulo + (uhi - ulo) * i / NT;
        double d1 = S.Bd * cbrt(curve_m2(u1)) / (2 * S.g), d2 = S.Bd * cbrt(curve_m2(u2)) / (2 * S.g);
        cum[i] = cum[i - 1] + 0.5 * (d1 + d2) * (u2 - u1);
    }
    double total = cum[NT];
    int K = max(1, cfg.chunks);
    vector<ld> bnd(K + 1);
    bnd[0] = ulo;
    bnd[K] = uhi;
    for (int j = 1; j < K; ++j) {
        double want = total * j / K;
        int idx = (int)(lower_bound(cum.begin(), cum.end(), want) - cum.begin());
        if (idx < 1) idx = 1;
        double f = (want - cum[idx - 1]) / max(1e-300, cum[idx] - cum[idx - 1]);
        bnd[j] = ulo + (uhi - ulo) * ((idx - 1) + f) / NT;
    }
    if (cfg.verbose)
        fprintf(stderr, "n=%s B=%lld zmin=%lld M=%d E=%.2f  u in [%.6f, %.6f]  expected boxes %.4g\n",
                str128(cfg.n).c_str(), cfg.B, cfg.zmin, cfg.M, cfg.E, ulo, uhi, total);

    Stats tot;
    vector<Hit> allhits;
    #pragma omp parallel
    {
        Stats st;
        vector<Hit> hits;
        #pragma omp for schedule(dynamic, 1)
        for (int j = 0; j < K; ++j) run_chunk(bnd[j], bnd[j + 1], S, sv, st, hits);
        #pragma omp critical
        {
            tot.add(st);
            allhits.insert(allhits.end(), hits.begin(), hits.end());
        }
    }
    double t1 = chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();

    sort(allhits.begin(), allhits.end(), [](const Hit& a, const Hit& b) {
        return tie(a.x, a.y, a.z) < tie(b.x, b.y, b.z);
    });
    allhits.erase(unique(allhits.begin(), allhits.end(), [](const Hit& a, const Hit& b) {
        return a.x == b.x && a.y == b.y && a.z == b.z;
    }), allhits.end());

    if (cfg.verbose)
        fprintf(stderr, "lattice part: %.3f s  boxes %lld (%.3g ns/box)  lll steps/box %.2f  leaves/box %.2f  in-cube/box %.2f  exact checks %lld  failures %lld\n",
                t1 - t0, tot.patches, (t1 - t0) * 1e9 / max(1LL, tot.patches), (double)tot.steps / max(1LL, tot.patches),
                (double)tot.leaves / max(1LL, tot.patches), (double)tot.boxed / max(1LL, tot.patches), tot.exact, tot.fail);

    if (cfg.near_mode) {
        for (auto& h : allhits) printf("%lld %lld %lld %s\n", h.x, h.y, h.z, str128(h.F).c_str());
        return tot.fail ? 2 : 0;
    }

    set<array<ll, 3>> sols;
    for (auto& h : allhits) {
        array<ll, 3> t = {h.x, h.y, -h.z};
        sort(t.begin(), t.end());
        sols.insert(t);
    }
    if (cfg.scan) {
        double s0 = chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
        small_scan(cfg.zmin, sols);
        double s1 = chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
        if (cfg.verbose) fprintf(stderr, "direct scan |a|,|b| <= %lld: %.3f s\n", cfg.zmin, s1 - s0);
    }
    printf("solutions of a^5+b^5+c^5 = %s with |a|,|b|,|c| <= %lld (sorted triples): %zu\n",
           str128(cfg.n).c_str(), cfg.B, sols.size());
    for (auto& t : sols) printf("%lld %lld %lld\n", t[0], t[1], t[2]);
    return tot.fail ? 2 : 0;
}
