"""Recompute steady states of Wilson & Cowan (1972) eqs. 11-12 with logistic eq. 15.
Used only to sanity-check the parameter sets read from the paper's figure captions."""
import numpy as np
from scipy.optimize import brentq

def S(x, a, th):
    return 1/(1+np.exp(-a*(x-th))) - 1/(1+np.exp(a*th))

def dS(x, a, th):
    s = 1/(1+np.exp(-a*(x-th)))
    return a*s*(1-s)

def kmax(a, th):  # sup of the shifted logistic = 1 - 1/(1+exp(a*th))
    return 1 - 1/(1+np.exp(a*th))

SETS = {
    "fig4_5": dict(c1=12, c2=4, c3=13, c4=11, ae=1.2, te=2.8, ai=1, ti=4),
    "fig6":   dict(c1=13, c2=4, c3=20, c4=2, ae=1.2, te=2.7, ai=5, ti=3.7),
    "fig7_8": dict(c1=13, c2=4, c3=22, c4=2, ae=1.5, te=2.5, ai=6, ti=4.3),
    "fig10":  dict(c1=15, c2=15, c3=15, c4=3, ae=1, te=2, ai=2, ti=2.5),
    "fig11_12": dict(c1=16, c2=12, c3=15, c4=3, ae=1.3, te=4, ai=2, ti=3.7),
}

def fixed_points(p, P=0.0, Q=0.0, kmode="max", re=1.0, ri=1.0, tau_ratio=1.0):
    ke = kmax(p["ae"], p["te"]) if kmode == "max" else 1.0
    ki = kmax(p["ai"], p["ti"]) if kmode == "max" else 1.0

    def I_of_E(E):
        g = lambda I: -I + (ki - ri*I)*S(p["c3"]*E - p["c4"]*I + Q, p["ai"], p["ti"])
        return brentq(g, -1.0, ki/ri - 1e-12)

    def f(E):
        I = I_of_E(E)
        return -E + (ke - re*E)*S(p["c1"]*E - p["c2"]*I + P, p["ae"], p["te"])

    Es = np.linspace(-0.3, ke/re - 1e-9, 4000)
    fs = np.array([f(E) for E in Es])
    roots = []
    for i in range(len(Es)-1):
        if fs[i] == 0 or fs[i]*fs[i+1] < 0:
            roots.append(brentq(f, Es[i], Es[i+1]))
    out = []
    for E in roots:
        I = I_of_E(E)
        xe = p["c1"]*E - p["c2"]*I + P
        xi = p["c3"]*E - p["c4"]*I + Q
        Se_, dSe = S(xe, p["ae"], p["te"]), dS(xe, p["ae"], p["te"])
        Si_, dSi = S(xi, p["ai"], p["ti"]), dS(xi, p["ai"], p["ti"])
        # tau_e = 1, tau_i = tau_ratio
        J = np.array([
            [-1 - re*Se_ + (ke - re*E)*dSe*p["c1"], -(ke - re*E)*dSe*p["c2"]],
            [((ki - ri*I)*dSi*p["c3"])/tau_ratio, (-1 - ri*Si_ - (ki - ri*I)*dSi*p["c4"])/tau_ratio],
        ])
        ev = np.linalg.eigvals(J)
        out.append((E, I, bool(np.all(ev.real < 0)), ev))
    return out

if __name__ == "__main__":
    for kmode in ["max", "one"]:
        print(f"\n===== k_e,k_i = {'S(+inf)' if kmode=='max' else '1'} =====")
        for name, p in SETS.items():
            P = 1.25 if name == "fig11_12" else 0.0
            fps = fixed_points(p, P=P, kmode=kmode)
            print(f"{name} (P={P}, Q=0): {len(fps)} fixed points")
            for E, I, st, ev in fps:
                print(f"   E={E:+.4f} I={I:+.4f} {'STABLE  ' if st else 'unstable'} eig={np.round(ev,3)}")
        # P sweep for hysteresis sets
        for name in ["fig4_5", "fig6", "fig7_8"]:
            p = SETS[name]
            Ps = np.round(np.arange(-0.8, 1.2001, 0.01), 3)
            nst = [sum(st for *_, st, _ in fixed_points(p, P=P, kmode=kmode)) for P in Ps]
            segs, cur = [], None
            for P, n in zip(Ps, nst):
                if cur is None or cur[0] != n:
                    cur = [n, P, P]; segs.append(cur)
                else:
                    cur[2] = P
            print(f"{name} P-sweep (Q=0), #stable states by P interval:",
                  "; ".join(f"[{a:+.2f},{b:+.2f}]:{n}" for n, a, b in segs))
    # tau ratio sensitivity at P=0 for fig4_5 and fig7_8
    print("\n===== tau_i/tau_e sensitivity (k = S(+inf)) =====")
    for name in ["fig4_5", "fig6", "fig7_8"]:
        for r in [0.5, 1.0, 2.0, 5.0]:
            fps = fixed_points(SETS[name], kmode="max", tau_ratio=r)
            print(name, "tau_i/tau_e =", r, ["S" if st else "U" for *_, st, _ in fps])
