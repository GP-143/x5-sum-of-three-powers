// =====================================================================================================
//  Elkies / Elsenhans-Jahnel "flagstone" lattice-reduction search for  a^5 + b^5 + c^5 = n   (n >= 1)
//  Finds ALL integer solutions with |a|,|b|,|c| <= N (default 1e8) in O(N) lattice reductions.
//
//  Idea (cone covering).  Put z = largest |coordinate| (>0), x' = -x, y' = -y.  For z >= R0 every solution
//  satisfies  x'^5 + y'^5 = z^5 - n  with 0 < x' <= y' <= z, i.e. (X,Y)=(x'/z,y'/z) lies within ~n/z^5 of the
//  curve Y=(1-X^5)^(1/5), X in [X_lo, 2^(-1/5)].  Cover the curve by "flagstones" (chord parallelograms); the
//  integer points of the double pyramid over a flagstone are the lattice points of a thin box K:
//        |z| <= N,  |x' - Xm z| <= W2,  |y' - sl x' - C z| <= W3,     vol(K) ~ Vt  (a few dozen lattice points)
//  They are enumerated after a (warm-started, incremental) 3-D LLL reduction in the scaled coordinates, and each
//  candidate is tested exactly.  Everything not covered (max|.| < R0) is found by a tiny direct search.
//
//  Fixed-n speed-up (new): x^5 == x (mod 2,3,5)  =>  a^5+b^5+c^5 == a+b+c (mod 30): a LINEAR congruence.  For a
//  fixed n only the coset {a+b+c == n mod 30} of Z^3 is searched (index-30 sublattice), so flagstones can be
//  30^(1/3) = 3.1x longer for the same work per flagstone.
// =====================================================================================================
#include <bits/stdc++.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace std;
typedef long double ld; typedef long long i64; typedef unsigned long long u64;
typedef __int128 i128; typedef unsigned __int128 u128;
static_assert(LDBL_MANT_DIG>=64, "needs x87 80-bit long double (x86-64 gcc/clang)");

// ----------------------------------------------------------------------------------------- parameters
static i64 N; static u64 n1, n2; static int M=1; static vector<int> RES;      // cosets: a+b+c == r (mod M), r in RES
static ld NL, invN, tau; static i64 R0; static double Vp=64, Vt;               // Vp = admissible points per flagstone
static const int FS=62;                                                        // fixed-point fraction bits
static const double BOXE=1.0+1e-6;

static inline ld i1282ld(i128 v){ bool neg=v<0; u128 a=neg?(u128)(-v):(u128)v;
    ld r=(ld)(u64)(a>>64)*18446744073709551616.0L+(ld)(u64)a; return neg?-r:r; }
static inline u64 pw5(u64 t){ u64 t2=t*t; return t2*t2*t; }
static inline u128 pw5w(i64 t){ u128 x=(u128)(i128)t; u128 s=x*x; return s*s*x; }

struct Flag{ ld X0,X1,L,Xm,sl,C,W2,W3; i64 slfp,Cfp,Xmfp; ld inv2,inv3; double detExp; double dXm,dC,dW2,dW3; };
struct Sol{ i64 a,b,c; i128 n;
    bool operator<(const Sol&o)const{ if(n!=o.n)return n<o.n; if(a!=o.a)return a<o.a; if(b!=o.b)return b<o.b; return c<o.c; }
    bool operator==(const Sol&o)const{ return a==o.a&&b==o.b&&c==o.c; } };
struct Stats{ u64 flags=0, pts=0, cold=0, lllit=0, ymax=0, fallback=0, outer=0;
    void add(const Stats&o){ flags+=o.flags; pts+=o.pts; cold+=o.cold; lllit+=o.lllit; ymax=max(ymax,o.ymax); fallback+=o.fallback; outer+=o.outer; } };
struct Ctx{ vector<Sol> out; Stats st; i64 b[3][3]; bool have=false; };

static inline ld M2of(ld X, ld Y){ ld Y2=Y*Y,Y4=Y2*Y2; return 4*X*X*X/(Y4*Y4*Y); }   // |Y''| for Y=(1-X^5)^(1/5)
static inline ld Yexact(ld X){ ld t=1-X*X*X*X*X; return powl(t,0.2L); }

// scaled coordinates (z/N, G2/W2, G3/W3) of the lattice vector v=(x',y',z); G2,G3 in exact 62-bit fixed point
static inline void image(const Flag&f,const i64 v[3],ld o[3]){
    o[0]=(ld)v[2]*invN;
    i128 g2=((i128)v[0]<<FS)-(i128)f.Xmfp*v[2];
    i128 g3=((i128)v[1]<<FS)-(i128)f.slfp*v[0]-(i128)f.Cfp*v[2];
    o[1]=i1282ld(g2)*f.inv2; o[2]=i1282ld(g3)*f.inv3;
}
template<class T> static inline T dot3(const T*a,const T*b){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
template<class T> static void lll3(i64 b[3][3], T u[3][3], Stats&st){          // LLL (delta=0.99), integer basis b, real images u
    const T delta=(T)0.99; T mu[3][3]={}, bs[3][3], bn[3];
    auto gs=[&](int i){ for(int d=0;d<3;++d) bs[i][d]=u[i][d];
        for(int j=0;j<i;++j){ mu[i][j]=dot3<T>(u[i],bs[j])/bn[j]; for(int d=0;d<3;++d) bs[i][d]-=mu[i][j]*bs[j][d]; }
        bn[i]=dot3<T>(bs[i],bs[i]); };
    gs(0); gs(1); int k=1, guard=0;
    while(k<3){
        for(int j=k-1;j>=0;--j){ T q=std::round(mu[k][j]);
            if(q!=0){ i64 qi=(i64)q; for(int d=0;d<3;++d){ b[k][d]-=qi*b[j][d]; u[k][d]-=q*u[j][d]; }
                for(int i=0;i<j;++i) mu[k][i]-=q*mu[j][i]; mu[k][j]-=q; } }
        if(bn[k] >= (delta-mu[k][k-1]*mu[k][k-1])*bn[k-1]){ ++k; if(k<3) gs(k); }
        else{ for(int d=0;d<3;++d){ swap(b[k][d],b[k-1][d]); swap(u[k][d],u[k-1][d]); }
            int kk=max(k-1,1); gs(0); for(int i=1;i<=kk;++i) gs(i); k=kk; }
        ++st.lllit; if(++guard>20000) break;
    }
}

// exact test of a candidate lattice vector (x',y',z): triple (a,b,c)=(-x',-y',z)
static inline void finish(Ctx&cx,i64 vx,i64 vy,i64 vz){
    i64 a=-vx,b=-vy,c=vz;
    if(max(llabs(a),max(llabs(b),llabs(c)))>N) return;
    if(a+b==0||b+c==0||a+c==0) return;                                  // trivial family (t,-t,m): reported separately
    ld est=(ld)a*a*a*a*a+(ld)b*b*b*b*b+(ld)c*c*c*c*c; if(fabsl(est)>1e35L) return;
    i128 m=(i128)(pw5w(a)+pw5w(b)+pw5w(c));                              // exact (|p5| < 2^127 certified by the estimate)
    if(m<(i128)n1||m>(i128)n2) return;
    i64 t[3]={a,b,c}; sort(t,t+3); cx.out.push_back({t[0],t[1],t[2],m});
}

// enumerate the lattice points of the cosets a+b+c==r (mod M), r in RES, inside the box |u_i|<=1 (u = scaled coordinates)
static void enumerate(const Flag&f, Ctx&cx, const double A[3][3], const double iv[3][3]){
    Stats&st=cx.st; const u64 span=n2-n1;
    double Rr[3]; for(int j=0;j<3;++j) Rr[j]=BOXE*(fabs(iv[j][0])+fabs(iv[j][1])+fabs(iv[j][2]));
    st.ymax=max<u64>(st.ymax,(u64)max(Rr[0],max(Rr[1],Rr[2])));
    int jin=0; if(Rr[1]>Rr[jin]) jin=1; if(Rr[2]>Rr[jin]) jin=2; const int jo1=(jin+1)%3, jo2=(jin+2)%3;   // longest range innermost
    double rinv[3]; for(int i=0;i<3;++i){ double a=A[i][jin]; if(fabs(a)<1e-14) a=(a<0?-1e-14:1e-14); rinv[i]=1.0/a; }
    const i64 *bi=cx.b[jin], *b1=cx.b[jo1], *b2=cx.b[jo2];
    const double a10=A[0][jo1],a11=A[1][jo1],a12=A[2][jo1],a20=A[0][jo2],a21=A[1][jo2],a22=A[2][jo2];
    for(int r: RES){
        double c[3]={ (double)r/(double)NL, -f.dXm*r/f.dW2, -f.dC*r/f.dW3 };                 // image of the coset representative (0,0,r)
        i64 k[3]; for(int j=0;j<3;++j) k[j]=llround(-(iv[j][0]*c[0]+iv[j][1]*c[1]+iv[j][2]*c[2]));   // Babai shift towards the box
        i64 w0[3]; for(int d=0;d<3;++d) w0[d]=(i64)(d==2?r:0)+k[0]*cx.b[0][d]+k[1]*cx.b[1][d]+k[2]*cx.b[2][d];
        ld cL[3]; image(f,w0,cL); c[0]=(double)cL[0]; c[1]=(double)cL[1]; c[2]=(double)cL[2];
        double yc[3]; for(int j=0;j<3;++j) yc[j]=-(iv[j][0]*c[0]+iv[j][1]*c[1]+iv[j][2]*c[2]);
        i64 lo1=(i64)ceil(yc[jo1]-Rr[jo1]), hi1=(i64)floor(yc[jo1]+Rr[jo1]), lo2=(i64)ceil(yc[jo2]-Rr[jo2]), hi2=(i64)floor(yc[jo2]+Rr[jo2]);
        if((double)(hi1-lo1+1)*(double)(hi2-lo2+1)>5e7){ fprintf(stderr,"internal error: enumeration range too large (basis not reduced)\n"); exit(2); }
        for(i64 t1=lo1;t1<=hi1;++t1){
            double s0=c[0]+t1*a10+lo2*a20, s1=c[1]+t1*a11+lo2*a21, s2=c[2]+t1*a12+lo2*a22;
            for(i64 t2=lo2;t2<=hi2;++t2,s0+=a20,s1+=a21,s2+=a22){
                ++st.outer;
                double u0=(-BOXE-s0)*rinv[0], v0=(BOXE-s0)*rinv[0]; if(u0>v0) swap(u0,v0);
                double u1=(-BOXE-s1)*rinv[1], v1=(BOXE-s1)*rinv[1]; if(u1>v1) swap(u1,v1);
                double u2=(-BOXE-s2)*rinv[2], v2=(BOXE-s2)*rinv[2]; if(u2>v2) swap(u2,v2);
                double lo=max(u0,max(u1,u2)), hi=min(v0,min(v1,v2)); if(lo>hi) continue;
                i64 ylo=(i64)ceil(lo), yhi=(i64)floor(hi); if(ylo>yhi) continue;
                i64 vx=w0[0]+t1*b1[0]+t2*b2[0]+ylo*bi[0], vy=w0[1]+t1*b1[1]+t2*b2[1]+ylo*bi[1], vz=w0[2]+t1*b1[2]+t2*b2[2]+ylo*bi[2];
                for(i64 y=ylo;y<=yhi;++y){
                    ++st.pts;
                    u64 P=pw5((u64)(-vx))+pw5((u64)(-vy))+pw5((u64)vz);                  // p5 mod 2^64 : window test, then exact check
                    if((u64)(P-n1)<=span) finish(cx,vx,vy,vz);
                    vx+=bi[0]; vy+=bi[1]; vz+=bi[2];
                }
            }
        }
    }
}

static void initBasis(Ctx&cx){ memset(cx.b,0,sizeof cx.b); cx.b[0][0]=1; cx.b[0][2]=1; cx.b[1][1]=1; cx.b[1][2]=1; cx.b[2][2]=M; cx.have=true; }  // basis of {a+b+c==0 mod M}

static void process(const Flag&f, Ctx&cx){
    Stats&st=cx.st; ++st.flags;
    bool cold=!cx.have; if(cold){ initBasis(cx); ++st.cold; }
    double A[3][3], iv[3][3];
    for(int attempt=0;;++attempt){
        ld uL[3][3]; for(int j=0;j<3;++j) image(f,cx.b[j],uL[j]);
        if(cold||attempt>0){ lll3<ld>(cx.b,uL,st); for(int j=0;j<3;++j)for(int i=0;i<3;++i) A[i][j]=(double)uL[j][i]; }
        else{ double ud[3][3]; for(int j=0;j<3;++j)for(int i=0;i<3;++i) ud[j][i]=(double)uL[j][i];
              lll3<double>(cx.b,ud,st); for(int j=0;j<3;++j)for(int i=0;i<3;++i) A[i][j]=ud[j][i]; }
        double det=A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])-A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])+A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
        iv[0][0]=(A[1][1]*A[2][2]-A[1][2]*A[2][1])/det; iv[0][1]=(A[0][2]*A[2][1]-A[0][1]*A[2][2])/det; iv[0][2]=(A[0][1]*A[1][2]-A[0][2]*A[1][1])/det;
        iv[1][0]=(A[1][2]*A[2][0]-A[1][0]*A[2][2])/det; iv[1][1]=(A[0][0]*A[2][2]-A[0][2]*A[2][0])/det; iv[1][2]=(A[0][2]*A[1][0]-A[0][0]*A[1][2])/det;
        iv[2][0]=(A[1][0]*A[2][1]-A[1][1]*A[2][0])/det; iv[2][1]=(A[0][1]*A[2][0]-A[0][0]*A[2][1])/det; iv[2][2]=(A[0][0]*A[1][1]-A[0][1]*A[1][0])/det;
        double rel=fabs(det)/f.detExp;
        if(rel>0.999&&rel<1.001) break;      // integrity check: covolume of the scaled lattice must be M/(N*W2*W3); otherwise redo in long double
        ++st.fallback; if(attempt>=1) break;
    }
    enumerate(f,cx,A,iv);
}

// arc length L with box volume 4 L N^2 (N h/2 + tau) = Vt,  h = |Y''| L^2/8   (cubic in L, Newton)
static inline ld solveL(ld X, ld Y, ld Lg){
    const ld N3=NL*NL*NL, N2=NL*NL; ld Lp=Lg>0?Lg:1e-9L;
    ld y2=Y*Y, y4=y2*y2; ld Yg=Y-(X*X*X*X)/y4*Lp; ld M2g=M2of(X+Lp,Yg>0?Yg:Y);
    ld a=M2g*N3/4, b=4*N2*tau, L=Lg>0?Lg:powl((ld)Vt/(a>0?a:1e-30L),1.0L/3);
    if(!(L>0)||!isfinite((double)L)) L=1e-3L;
    for(int it=0;it<(Lg>0?3:8);++it){ ld fv=a*L*L*L+b*L-(ld)Vt, fp=3*a*L*L+b; L-=fv/fp; if(L<=0) L=1e-12L; }
    return min<ld>(L,0.05L);
}
static ld cubicL(ld X, ld Lg){ return solveL(X,Yexact(X),Lg); }

static void run_chunk(ld Xa, ld Xb, Ctx&cx){
    cx.have=false; ld X0=Xa, Y0=Yexact(X0), Lg=0; const ld N2=NL*NL;
    static const ld S2=4611686018427387904.0L, iS=1.0L/4611686018427387904.0L;      // 2^62
    while(X0<Xb){
        ld L=solveL(X0,Y0,Lg), X1, Y1, M2, h;
        ld y02=Y0*Y0, y04=y02*y02, X02=X0*X0, Yp0=-(X02*X02)/y04, M20=4*X02*X0/(y04*y04*Y0);
        for(int tries=0;;++tries){
            if(X0+L>Xb) L=Xb-X0; X1=X0+L;
            ld Y1g=Y0+Yp0*L-M20*L*L/2;                                              // Taylor guess, one Newton step on Y^5 = 1-X^5
            ld T=1-X1*X1*X1*X1*X1; Y1=Y1g; { ld y2=Y1*Y1,y4=y2*y2; Y1-=(y4*Y1-T)/(5*y4); }
            M2=M2of(X1,Y1); h=M2*L*L/8; ld vol=4*L*N2*(NL*h/2+tau);
            if(vol<=1.5L*Vt||tries>20||L<1e-15L) break; L*=0.8L; }
        Flag f; f.X0=X0; f.X1=X1; f.L=L; f.Xm=(X0+X1)/2; ld Ym=(Y0+Y1)/2; f.sl=(Y1-Y0)/L; ld sigma=h/2;   // chord through the arc ends, shifted by h/2
        f.C=Ym-f.sl*f.Xm+sigma;
        ld marg=16*NL*LDBL_EPSILON+1e-15L;                                            // covers long-double + coefficient rounding
        f.W2=L*NL/2*(1+1e-9L); f.W3=NL*h/2*(1+1e-9L)+tau+marg;
        f.slfp=(i64)llroundl(f.sl*S2); f.Cfp=(i64)llroundl(f.C*S2); f.Xmfp=(i64)llroundl(f.Xm*S2);
        f.inv2=iS/f.W2; f.inv3=iS/f.W3; f.detExp=(double)((ld)M/(NL*f.W2*f.W3));
        f.dXm=(double)f.Xm; f.dC=(double)f.C; f.dW2=(double)f.W2; f.dW3=(double)f.W3;
        process(f,cx);
        X0=X1; Y0=Y1; Lg=L;
    }
}

// all a<=b<=c with |.|<=R and n1<=p5<=n2 (two-pointer over sorted fifth powers), trivial family excluded
static void direct(i64 R, vector<Sol>&out){
    vector<i128> f(2*R+1); for(i64 t=-R;t<=R;++t) f[t+R]=(i128)(pw5w(t));
    #pragma omp parallel
    { vector<Sol> loc;
      #pragma omp for schedule(dynamic,8)
      for(i64 ci=0;ci<=2*R;++ci){ i64 c=ci-R; size_t p=0; bool init=false;
        for(i64 b=-R;b<=c;++b){
            i128 T=f[ci]+f[b+R]; i128 lo=(i128)n1-T, hi=(i128)n2-T;
            if(!init){ p=lower_bound(f.begin(),f.end(),lo)-f.begin(); init=true; }
            else{ while(p>0&&f[p-1]>=lo) --p; while(p<f.size()&&f[p]<lo) ++p; }
            for(size_t q=p;q<=(size_t)(b+R)&&q<f.size()&&f[q]<=hi;++q){ i64 a=(i64)q-R; if(a+b==0||b+c==0||a+c==0) continue; loc.push_back({a,b,c,T+f[q]}); } } }
      #pragma omp critical
      out.insert(out.end(),loc.begin(),loc.end()); }
}

// ------------------------------------------------------------------------------------------ engine entry
struct Params{ i64 N; u64 n1,n2; int M; vector<int> RES; double Vp=64; i64 R0override=0; };
struct Result{ vector<Sol> sols; Stats st; double t_direct=0,t_flag=0; i64 R0=0; double estFlags=0; int threads=1; };

static Result run_engine(const Params&p){
    N=p.N; n1=p.n1; n2=p.n2; M=p.M; RES=p.RES; Vp=p.Vp; NL=(ld)N; invN=1/NL; Vt=Vp*M/RES.size();
    // R0: smallest height handled by flagstones: n-offset term tau at z=R0 kept <= 25% of the curvature half-width at z=N
    { ld X=0.6L, Y=Yexact(X), M2=M2of(X,Y), L=powl(4*(ld)Vt/(M2*NL*NL*NL),1.0L/3), h=M2*L*L/8, Nr=NL*h/2;
      ld r=powl((ld)n2/(2.8L*0.25L*Nr),0.25L); R0=(i64)ceill(r); i64 lowR=(i64)(2.2L*powl((ld)n2,0.25L))+10; if(R0<lowR) R0=lowR;
      if(p.R0override) R0=p.R0override; }
    tau=(ld)n2/(2.8L*(ld)R0*R0*R0*R0);
    Result res; res.R0=R0; auto t0=chrono::steady_clock::now();
    direct(min<i64>(R0,N),res.sols);
    auto t1=chrono::steady_clock::now();
    if(R0<N){
        // non-trivial solutions with z>=R0 have X'=x'/z >= (1-n/z^4)^(1/5) z^(-1/5) >= 0.95 N^(-1/5);  x'<=y' gives X' <= 2^(-1/5)(1+eps)
        ld Xlo=0.95L*powl(NL,-0.2L), Xhi=powl(0.5L,0.2L)*(1+1e-9L+2*(ld)n2/((ld)R0*R0*R0*R0*R0));
        int nth=1;
#ifdef _OPENMP
        nth=omp_get_max_threads();
#endif
        res.threads=nth;
        const int G=20000; vector<ld> cum(G+1,0); { ld Lg=0; for(int g=0;g<G;++g){ ld X=Xlo+(Xhi-Xlo)*(g+0.5L)/G; ld L=cubicL(X,Lg); Lg=L; cum[g+1]=cum[g]+(Xhi-Xlo)/G/L; } }
        res.estFlags=(double)cum[G];
        int nch=(nth==1)?1:nth*24; vector<ld> bnd(nch+1); bnd[0]=Xlo; bnd[nch]=Xhi;      // chunks of equal expected work
        for(int c=1;c<nch;++c){ ld target=cum[G]*c/nch; int g=lower_bound(cum.begin(),cum.end(),target)-cum.begin(); if(g<1)g=1; if(g>G)g=G;
            ld frac=(target-cum[g-1])/(cum[g]-cum[g-1]); bnd[c]=Xlo+(Xhi-Xlo)*((g-1)+frac)/G; }
        #pragma omp parallel for schedule(dynamic,1)
        for(int c=0;c<nch;++c){ Ctx cx; run_chunk(bnd[c],bnd[c+1],cx);
            #pragma omp critical
            { res.sols.insert(res.sols.end(),cx.out.begin(),cx.out.end()); res.st.add(cx.st); } }
    }
    auto t2=chrono::steady_clock::now();
    sort(res.sols.begin(),res.sols.end()); res.sols.erase(unique(res.sols.begin(),res.sols.end()),res.sols.end());
    res.t_direct=chrono::duration<double>(t1-t0).count(); res.t_flag=chrono::duration<double>(t2-t1).count();
    return res;
}

// ------------------------------------------------------------------------------------------ small helpers
static inline bool allowed11(u64 n){ int r=(int)(n%11); return !(r>=4&&r<=7); }      // a^5,b^5,c^5 in {0,+1,-1} (mod 11) => sum in {-3..3}
static bool is_fifth_power(u64 n, u64&m){ u64 r=(u64)llroundl(powl((ld)n,0.2L)); for(u64 t=(r>2?r-2:0);t<=r+2;++t){ u128 x=t; if(x*x*x*x*x==(u128)n){ m=t; return true; } } return false; }

// brute-force reference (O(N^2)); used only by --selftest
static vector<Sol> brute_ref(i64 Nb,u64 lo,u64 hi){
    vector<Sol> v; vector<i128> f(2*Nb+1); for(i64 t=-Nb;t<=Nb;++t) f[t+Nb]=(i128)pw5w(t);
    for(i64 c=-Nb;c<=Nb;++c) for(i64 b=-Nb;b<=c;++b){ i128 T=f[c+Nb]+f[b+Nb];
        double l=(double)((i128)lo-T), h=(double)((i128)hi-T); double rl=l<0?-pow(-l,0.2):pow(l,0.2), rh=h<0?-pow(-h,0.2):pow(h,0.2);
        i64 alo=(i64)floor(rl)-1, ahi=(i64)ceil(rh)+1; if(alo<-Nb) alo=-Nb; if(ahi>b) ahi=b;
        for(i64 a=alo;a<=ahi;++a){ i128 m=T+f[a+Nb]; if(m>=(i128)lo&&m<=(i128)hi&&a+b!=0&&b+c!=0&&a+c!=0) v.push_back({a,b,c,m}); } }
    sort(v.begin(),v.end()); v.erase(unique(v.begin(),v.end()),v.end()); return v;
}
static int selftest(){
    printf("self-test: mod-11 table ... ");
    { set<int> s; for(int a=0;a<11;++a)for(int b=0;b<11;++b)for(int c=0;c<11;++c){ int x=1,y=1,z=1; for(int e=0;e<5;++e){x=x*a%11;y=y*b%11;z=z*c%11;} s.insert((x+y+z)%11); }
      bool ok=true; for(int r=0;r<11;++r) ok&= (s.count(r)>0)==(r<4||r>7); printf("%s\n",ok?"ok (4,5,6,7 impossible)":"FAIL"); if(!ok) return 1; }
    printf("self-test: x^5 == x (mod 30) ... "); { bool ok=true; for(int x=-60;x<=60;++x){ long long p=(long long)x*x*x*x*x; ok&=(((p-x)%30)==0); } printf("%s\n",ok?"ok":"FAIL"); if(!ok) return 1; }
    const i64 Nb=2200; int bad=0;
    struct T{ u64 lo,hi; int M; vector<int> R; const char*nm; };
    vector<T> tests={ {1,1500000000ULL,1,{0},"window [1,1.5e9], plain lattice"}, {400000000ULL,900000000ULL,1,{0},"window [4e8,9e8], plain lattice"},
                      {1,1500000000ULL,30,{7},"coset a+b+c=7 (mod 30)"}, {1,1500000000ULL,30,{0,1,2},"cosets {0,1,2} (mod 30)"}, {1,1500000000ULL,6,{1,5},"cosets {1,5} (mod 6)"}, {1,1500000000ULL,30,{15},"coset 15 (mod 30)"} };
    for(auto&t:tests){ Params p; p.N=Nb; p.n1=t.lo; p.n2=t.hi; p.M=t.M; p.RES=t.R; Result r=run_engine(p);
        auto ref=brute_ref(Nb,t.lo,t.hi);
        // direct part is not residue-filtered: compare everything for the plain lattice, and only max>R0 for coset tests
        auto keep=[&](const Sol&s){ if(t.M==1) return true; if(max(llabs(s.a),max(llabs(s.b),llabs(s.c)))<=r.R0) return false; int rr=(int)(s.n%t.M); for(int q:t.R) if(q==rr) return true; return false; };
        vector<Sol> A,B; for(auto&s:ref) if(keep(s)) A.push_back(s); for(auto&s:r.sols) if(keep(s)) B.push_back(s);
        size_t hiCnt=0; for(auto&s:A) if(max(llabs(s.a),max(llabs(s.b),llabs(s.c)))>r.R0) ++hiCnt;
        bool ok=(A==B); printf("self-test: N=%lld %-34s R0=%lld  brute=%zu engine=%zu (flagstone-region solutions: %zu)  %s\n",(long long)Nb,t.nm,(long long)r.R0,A.size(),B.size(),hiCnt,ok?"PASS":"FAIL"); if(!ok) ++bad; }
    // fixed-n mode (single mod-30 coset) on real solutions found by brute force
    { auto ref=brute_ref(Nb,1,1500000000ULL); vector<Sol> hi; for(auto&s:ref) if(max(llabs(s.a),max(llabs(s.b),llabs(s.c)))>1100) hi.push_back(s);
      size_t tested=0, found=0; for(size_t i=0;i<hi.size()&&tested<25;i+=max<size_t>(1,hi.size()/25)){ Params p; p.N=Nb; p.n1=p.n2=(u64)hi[i].n; p.M=30; p.RES={(int)(hi[i].n%30)}; Result r=run_engine(p);
          ++tested; for(auto&s:r.sols) if(s==hi[i]) {++found;break;} }
      printf("self-test: fixed-n (single coset) on %zu known solutions with height>1100: found %zu  %s\n",tested,found,found==tested?"PASS":"FAIL"); if(found!=tested) ++bad; }
    printf(bad?"SELF-TEST FAILED\n":"ALL SELF-TESTS PASSED\n"); return bad?1:0;
}

// =====================================================================================================
//  quintic_fixed :  all integer solutions of  a^5+b^5+c^5 = n   (n >= 1 fixed),  |a|,|b|,|c| <= N
//  usage:  ./quintic_fixed n [N=100000000] [Vp=64]        |   ./quintic_fixed --selftest
// =====================================================================================================
int main(int argc,char**argv){
    if(argc>=2 && string(argv[1])=="--selftest") return selftest();
    if(argc<2){ fprintf(stderr,"usage: %s n [N=100000000] [Vp=64]   |   %s --selftest\n",argv[0],argv[0]); return 1; }
    u64 n=strtoull(argv[1],0,10); i64 Nb=argc>2?atoll(argv[2]):100000000LL; double V=argc>3?atof(argv[3]):64.0;
    if(n<1||n>=(1ULL<<62)){ fprintf(stderr,"need 1 <= n < 2^62\n"); return 1; }
    if(Nb<10||Nb>300000000LL){ fprintf(stderr,"need 10 <= N <= 3e8 (long-double precision limit of the flagstone boxes)\n"); return 1; }
    printf("a^5+b^5+c^5 = %llu,  |a|,|b|,|c| <= %lld\n",(unsigned long long)n,(long long)Nb);
    if(!allowed11(n)){ printf("n = %llu == %d (mod 11): NO solutions over Z (a^5,b^5,c^5 are 0,+1,-1 mod 11, so the sum is in {-3..3} mod 11).\n",(unsigned long long)n,(int)(n%11)); return 0; }
    u64 m; if(is_fifth_power(n,m)) printf("note: n = %llu^5 -> trivial infinite family (t,-t,%llu), t arbitrary; omitted below.\n",(unsigned long long)m,(unsigned long long)m);
    Params p; p.N=Nb; p.n1=p.n2=n; p.M=30; p.RES={(int)(n%30)}; p.Vp=V;           // single coset  a+b+c == n (mod 30)
    Result r=run_engine(p);
    for(auto&s:r.sols) printf("SOLUTION  %lld^5 + %lld^5 + %lld^5 = %llu\n",(long long)s.a,(long long)s.b,(long long)s.c,(unsigned long long)n);
    if(r.sols.empty()) printf("no non-trivial solution with |a|,|b|,|c| <= %lld\n",(long long)Nb);
    fprintf(stderr,"[R0=%lld  flagstones=%llu (est %.0f)  box points/flagstone=%.1f  LLL fallbacks=%llu  threads=%d  direct %.2fs  lattice %.2fs]\n",
        (long long)r.R0,(unsigned long long)r.st.flags,r.estFlags,r.st.flags?(double)r.st.pts/r.st.flags:0.0,(unsigned long long)r.st.fallback,r.threads,r.t_direct,r.t_flag);
    return 0;
}
