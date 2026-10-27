"""Baseline reliability methods for the seven mathematical examples.

New baselines (same 1200 true-evaluation budget as the paper):
  - mcs:        direct Monte Carlo
  - akmcs:      GP-based AK-MCS in the full input space (scalar system margin)
  - as_gp_mcs:  active subspace + GP + MCS
  - as_akmcs:   active subspace + AK-MCS in the reduced space

Gradients used by the active-subspace baselines are analytic and counted as
one model call per gradient sample (200 samples). This is a deliberately
strong, gradient-enabled baseline; the finite-difference cost is not used.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern

from method_benchmarks import Problem

ROOT = Path(__file__).resolve().parent
B = 1200
N_GRAD = 200
BATCH = 40
POOL = 20000


def system_margin(p, g):
    """Scalar margin whose nonpositive set is the system failure event."""
    if p.number == 7:
        return np.minimum(np.maximum(g[:, 0], g[:, 1]), g[:, 2])
    return np.min(g, axis=1)


def gradient_u(p, u):
    """Analytic gradients of all component margins w.r.t. u.

    z = u @ B, so dg/du = B @ dg/dz.  Returns (N, p, d).
    """
    z = u @ p.basis
    n = p.number
    N = len(u)
    p_out = p.p
    r = p.r
    Gz = np.zeros((N, p_out, r))
    if n == 1:
        for k in range(4):
            Gz[:, k, k] = -1.0
    elif n == 2:
        d = z[:, 0] - z[:, 1]
        Gz[:, 0, 0] = -1.0 / np.sqrt(2.0) + 0.2 * d
        Gz[:, 0, 1] = -1.0 / np.sqrt(2.0) - 0.2 * d
    elif n == 3:
        d = z[:, 0] - z[:, 1]
        Gz[:, 0, 0] = 0.2 * d - 1.0 / np.sqrt(2.0)
        Gz[:, 0, 1] = -0.2 * d - 1.0 / np.sqrt(2.0)
        Gz[:, 1, 0] = 0.2 * d + 1.0 / np.sqrt(2.0)
        Gz[:, 1, 1] = -0.2 * d + 1.0 / np.sqrt(2.0)
        Gz[:, 2, 0] = 1.0
        Gz[:, 2, 1] = -1.0
        Gz[:, 3, 0] = -1.0
        Gz[:, 3, 1] = 1.0
    elif n == 4:
        c = np.array([0.7, -0.7, 0.5, -0.5])
        for k in range(4):
            Gz[:, k, 0] = -c[k]
            Gz[:, k, 1 + k] = -0.7 - 0.3 * z[:, 1 + k]
    elif n == 5:
        c1 = np.array([3.5, 0.0])
        c2 = np.array([-4.1, 0.3])
        Gz[:, 0, :] = 2.0 * (z - c1)
        Gz[:, 1, :] = 2.0 * (z - c2)
    elif n == 6:
        for k in range(2):
            sl = slice(k * 10, (k + 1) * 10)
            Gz[:, k, sl] = -2.0 * z[:, sl]
    elif n == 7:
        a, b = z[:, 0], z[:, 1]
        Gz[:, 0, 0] = -2.0 * a
        Gz[:, 0, 1] = 3.0 * b * b
        Gz[:, 1, 0] = -1.0
        Gz[:, 1, 1] = -8.0
        Gz[:, 2, 0] = 2.0 * (a + 3.0)
        Gz[:, 2, 1] = 2.0 * (b + 3.0)
    else:
        raise ValueError(n)
    return np.einsum("ij,npj->npi", p.basis, Gz)


def active_subspace(p, rng, n_grad=N_GRAD, energy=0.99, cap=12):
    u = rng.normal(size=(n_grad, p.d))
    g = p.evaluate(u)
    sd = np.maximum(g.std(axis=0), 1e-8)
    G = gradient_u(p, u) / sd[None, :, None]
    Cmat = np.einsum("npi,npj->ij", G, G) / n_grad
    w, V = np.linalg.eigh(Cmat)
    order = np.argsort(w)[::-1]
    w = np.maximum(w[order], 0.0)
    V = V[:, order]
    total = w.sum()
    if total <= 0:
        r = 1
    else:
        r = int(np.searchsorted(np.cumsum(w) / total, energy) + 1)
    r = max(1, min(r, cap, p.d))
    return V[:, :r], r, w


def gp_fit(X, y, ls):
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=ls, nu=2.5)
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None,
                                  alpha=1e-6, normalize_y=True)
    gp.fit(X, y)
    return gp


def gp_predict(gp, X, chunk=20000):
    mu = []
    sd = []
    for k in range(0, len(X), chunk):
        m, s = gp.predict(X[k:k + chunk], return_std=True)
        mu.append(m)
        sd.append(s)
    return np.concatenate(mu), np.concatenate(sd)


def mcs(p, rng, budget=B):
    u = rng.normal(size=(budget, p.d))
    g = p.evaluate(u)
    y = p.event(g)
    return dict(Pf=float(y.mean()), calls=budget, rank=p.d,
                extra=dict(method_detail="direct Monte Carlo"))


def akmcs(p, rng, budget=B, ls=10.0):
    n0 = 20
    X = rng.normal(size=(n0, p.d))
    y = system_margin(p, p.evaluate(X))
    fits = 0
    while len(X) < budget:
        gp = gp_fit(X, y, ls)
        fits += 1
        cand = rng.normal(size=(POOL, p.d))
        mu, sd = gp_predict(gp, cand, chunk=10000)
        U = np.abs(mu) / np.maximum(sd, 1e-12)
        take = min(BATCH, budget - len(X))
        idx = np.argsort(U)[:take]
        Xn = cand[idx]
        yn = system_margin(p, p.evaluate(Xn))
        X = np.vstack([X, Xn])
        y = np.concatenate([y, yn])
    return dict(Pf=None, calls=len(X), rank=p.d,
                extra=dict(method_detail="AK-MCS full space", fits=fits,
                           initial=n0, batch=BATCH, pool=POOL))


def as_gp_mcs(p, rng, budget=B, n_grad=N_GRAD):
    U, r, eig = active_subspace(p, rng, n_grad)
    n_lab = budget - n_grad
    X = rng.normal(size=(n_lab, p.d))
    y = system_margin(p, p.evaluate(X))
    Z = X @ U
    gp = gp_fit(Z, y, ls=1.0)
    return dict(Pf=None, calls=budget, rank=r,
                extra=dict(method_detail="AS + GP + MCS", n_grad=n_grad,
                           n_labels=n_lab, energy=float(np.cumsum(eig)[r - 1] / eig.sum())))


def as_akmcs(p, rng, budget=B, n_grad=N_GRAD):
    U, r, eig = active_subspace(p, rng, n_grad)
    n_lab = budget - n_grad
    n0 = 20
    X = rng.normal(size=(n0, p.d))
    y = system_margin(p, p.evaluate(X))
    Z = X @ U
    fits = 0
    while len(Z) < n_lab:
        gp = gp_fit(Z, y, ls=1.0)
        fits += 1
        cand_z = rng.normal(size=(POOL, r))
        mu, sd = gp_predict(gp, cand_z, chunk=10000)
        Uf = np.abs(mu) / np.maximum(sd, 1e-12)
        take = min(BATCH, n_lab - len(Z))
        idx = np.argsort(Uf)[:take]
        Zn = cand_z[idx]
        # lift to a full input with the requested retained coordinates
        resid = rng.normal(size=(take, p.d))
        resid -= (resid @ U) @ U.T
        Xn = Zn @ U.T + resid
        yn = system_margin(p, p.evaluate(Xn))
        Z = np.vstack([Z, Zn])
        y = np.concatenate([y, yn])
    return dict(Pf=None, calls=budget, rank=r,
                extra=dict(method_detail="AS + AK-MCS", n_grad=n_grad,
                           n_labels=n_lab, fits=fits, initial=n0, batch=BATCH,
                           pool=POOL, energy=float(np.cumsum(eig)[r - 1] / eig.sum())))


def evaluate_surrogate(p, gp, U, cloud, reduced):
    if reduced:
        Z = cloud @ U
        mu, _ = gp_predict(gp, Z, chunk=20000)
    else:
        mu, _ = gp_predict(gp, cloud, chunk=20000)
    return float(np.mean(mu <= 0.0))


def run_case(p, methods, seed):
    n = p.number
    pref, qerr = p.reference()
    rng = np.random.default_rng(seed)
    cloud = ndtri(np.clip(qmc.Sobol(p.d, scramble=True, seed=72000 + n)
                          .random_base2(18), 1e-12, 1 - 1e-12))
    out = {}
    for method in methods:
        rng_m = np.random.default_rng(seed)
        t0 = time.perf_counter()
        if method == "mcs":
            res = mcs(p, rng_m)
        elif method == "akmcs":
            res = akmcs(p, rng_m)
        elif method == "as_gp_mcs":
            res = as_gp_mcs(p, rng_m)
        elif method == "as_akmcs":
            res = as_akmcs(p, rng_m)
        else:
            raise ValueError(method)
        # Recover the fitted surrogate for probability evaluation.
        # The functions above return None for Pf; refit here cheaply from
        # the same data is avoided by storing the predictor in the result.
        res["seconds"] = time.perf_counter() - t0
        res["reference"] = pref
        res["quadrature_error"] = qerr
        out[method] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="1,2,3,4,5,6,7")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--methods", default="mcs,akmcs,as_gp_mcs,as_akmcs")
    ap.add_argument("--output", default="results/method_paper_v2/baselines.json")
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()
    dest = ROOT / args.output
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(dest.read_text(encoding="utf-8")) if dest.exists() else {"runs": []}
    methods = args.methods.split(",")
    as_ls = 5.0
    for n in map(int, args.cases.split(",")):
        p = Problem(n)
        pref, qerr = p.reference()
        cloud = ndtri(np.clip(qmc.Sobol(p.d, scramble=True, seed=72000 + n)
                              .random_base2(18), 1e-12, 1 - 1e-12))
        for s in range(args.start, args.start + args.seeds):
            seed = 410000 + n * 1000 + s
            key = f"{n}:{s}"
            if any(r["case"] == n and r["seed_index"] == s and r["method"] in methods
                   for r in data["runs"]):
                # only skip exact method entries
                pass
            for method in methods:
                if any(r["case"] == n and r["seed_index"] == s and r["method"] == method
                       for r in data["runs"]):
                    continue
                print(f"case={n} seed={s} method={method}", flush=True)
                rng = np.random.default_rng(seed)
                t0 = time.perf_counter()
                # The run functions need the predictor; inline the method-specific
                # probability calculation here to keep the JSON self-contained.
                if method == "mcs":
                    u = rng.normal(size=(B, p.d))
                    g = p.evaluate(u)
                    y = p.event(g)
                    res = dict(Pf=float(y.mean()), calls=B, rank=p.d,
                               extra=dict(method_detail="direct Monte Carlo"))
                elif method == "akmcs":
                    # Fit and evaluate in one place.
                    X = rng.normal(size=(20, p.d))
                    yy = system_margin(p, p.evaluate(X))
                    fits = 0
                    while len(X) < B:
                        gp = gp_fit(X, yy, ls=10.0)
                        fits += 1
                        cand = rng.normal(size=(POOL, p.d))
                        mu, sd = gp_predict(gp, cand, chunk=10000)
                        Uf = np.abs(mu) / np.maximum(sd, 1e-12)
                        take = min(BATCH, B - len(X))
                        idx = np.argsort(Uf)[:take]
                        Xn = cand[idx]
                        yn = system_margin(p, p.evaluate(Xn))
                        X = np.vstack([X, Xn])
                        yy = np.concatenate([yy, yn])
                    gp = gp_fit(X, yy, ls=10.0)
                    mu, _ = gp_predict(gp, cloud, chunk=10000)
                    res = dict(Pf=float(np.mean(mu <= 0.0)), calls=len(X), rank=p.d,
                               extra=dict(method_detail="AK-MCS full space", fits=fits,
                                          initial=20, batch=BATCH, pool=POOL))
                elif method == "as_gp_mcs":
                    U, r, eig = active_subspace(p, rng, N_GRAD)
                    n_lab = B - N_GRAD
                    X = rng.normal(size=(n_lab, p.d))
                    yy = system_margin(p, p.evaluate(X))
                    gp = gp_fit(X @ U, yy, ls=as_ls)
                    res = dict(Pf=evaluate_surrogate(p, gp, U, cloud, True),
                               calls=B, rank=r,
                               extra=dict(method_detail="AS + GP + MCS",
                                          n_grad=N_GRAD, n_labels=n_lab,
                                          energy=float(np.cumsum(eig)[r - 1] / eig.sum())))
                elif method == "as_akmcs":
                    U, r, eig = active_subspace(p, rng, N_GRAD)
                    n_lab = B - N_GRAD
                    X = rng.normal(size=(20, p.d))
                    yy = system_margin(p, p.evaluate(X))
                    Z = X @ U
                    fits = 0
                    while len(Z) < n_lab:
                        gp = gp_fit(Z, yy, ls=as_ls)
                        fits += 1
                        cand_z = rng.normal(size=(POOL, r))
                        mu, sd = gp_predict(gp, cand_z, chunk=10000)
                        Uf = np.abs(mu) / np.maximum(sd, 1e-12)
                        take = min(BATCH, n_lab - len(Z))
                        idx = np.argsort(Uf)[:take]
                        Zn = cand_z[idx]
                        resid = rng.normal(size=(take, p.d))
                        resid -= (resid @ U) @ U.T
                        Xn = Zn @ U.T + resid
                        yn = system_margin(p, p.evaluate(Xn))
                        Z = np.vstack([Z, Zn])
                        yy = np.concatenate([yy, yn])
                    gp = gp_fit(Z, yy, ls=as_ls)
                    res = dict(Pf=evaluate_surrogate(p, gp, U, cloud, True),
                               calls=B, rank=r,
                               extra=dict(method_detail="AS + AK-MCS",
                                          n_grad=N_GRAD, n_labels=n_lab,
                                          fits=fits, initial=20, batch=BATCH,
                                          pool=POOL,
                                          energy=float(np.cumsum(eig)[r - 1] / eig.sum())))
                res.update(dict(case=n, seed_index=s, seed=seed, method=method,
                                seconds=time.perf_counter() - t0,
                                reference=p.reference()[0]))
                data["runs"].append(res)
                dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
                print("  Pf=%.6g ref=%.6g rank=%d time=%.1f" %
                      (res["Pf"], res["reference"], res["rank"], res["seconds"]),
                      flush=True)
    print("done", dest)


if __name__ == "__main__":
    main()
