"""Parsimonious polynomial experts for CC-DAS.

The representation and label budget come from the v3 validation-gated CC-DAS
trajectory. On the same labelled data, this module fits a small ladder of
polynomial response maps (degree 1/2/3 on learned subspaces of several ranks),
plus the neural expert, and selects the simplest model whose held-out
unweighted MSE is within one standard error of the best. No extra model calls
are used.
"""
import numpy as np
import torch
from torch import nn

from reproducibility import seed_everything


class PolyNet(nn.Module):
    def __init__(self, d, p, m, degree, seed, basis=None):
        super().__init__()
        seed_everything(seed, True)
        self.d, self.p, self.m, self.degree = d, p, m, degree
        self.raw = nn.Parameter(torch.tensor(basis, dtype=torch.float32)
                                if basis is not None
                                else torch.randn(d, m) * 0.1)
        exps = []
        for deg in range(1, degree + 1):
            def rec(prefix, left, start):
                if left == 0:
                    exps.append(tuple(prefix)); return
                for k in range(start, m):
                    rec(prefix + [k], left - 1, k)
            rec([], deg, 0)
        self.exps = exps
        self.coef = nn.Parameter(torch.randn(p, len(exps)) * 0.1)
        self.a = nn.Parameter(torch.zeros(p))

    def basis(self):
        q, r = torch.linalg.qr(self.raw, mode='reduced')
        sign = torch.where(torch.diagonal(r) >= 0, 1., -1.)
        return q * sign

    def forward(self, u):
        z = u @ self.basis()
        feats = []
        for e in self.exps:
            f = torch.ones(z.shape[0], device=z.device)
            for k in e:
                f = f * z[:, k]
            feats.append(f)
        return self.a + torch.stack(feats, 1) @ self.coef.T


def train_poly(net, x, g, mu, sd, seed, epochs=1500):
    seed_everything(seed, True)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor((g - mu) / sd, dtype=torch.float32)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)
    for _ in range(epochs):
        loss = ((net(xt) - yt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()


def predict(net, x, mu, sd):
    with torch.no_grad():
        return net(torch.tensor(x, dtype=torch.float32)).numpy() * sd + mu


def fit_candidates(x, g, xv, gv, mu, sd, W0, p, mlp_score, seed_base,
                   grid=None, epochs=1500, dual_init=False, random_ranks=None):
    """Fit the polynomial ladder and return candidate records.

    Each record has degree, rank, holdout MSE, standard error, and a net (or
    None for the neural candidate). With dual_init=True, every candidate is
    fitted once from the MLP basis (warm) and once from a random basis.
    """
    if grid is None:
        grid = {1: [2, 4, 5, 8, 12, 20],
                2: [2, 4, 5, 8, 12, 20],
                3: [2, 4]}
    cands = [dict(name='mlp', degree=0, rank=W0.shape[1],
                  mse=mlp_score, se=0.0, net=None)]
    for degree, ranks in grid.items():
        for r in ranks:
            if r > p.d or (degree == 3 and r > 5):
                continue
            if random_ranks is not None:
                inits = ['warm', 'random'] if r in random_ranks else ['warm']
            else:
                inits = ['warm', 'random'] if dual_init else ['warm']
            for init in inits:
                keep = min(r, W0.shape[1])
                if init == 'warm':
                    B = np.column_stack([
                        W0[:, :keep],
                        np.random.default_rng(seed_base + 100 * degree + r)
                        .normal(size=(p.d, r - keep)),
                    ]) if r > keep else W0[:, :r]
                    suffix = ''
                else:
                    B = None
                    suffix = 'R'
                seed = seed_base + 100 * degree + r + (0 if init == 'warm' else 500)
                net = PolyNet(p.d, p.p, r, degree, seed,
                              basis=None if B is None else B[:, :r])
                train_poly(net, x, g, mu, sd, seed, epochs)
                err = np.mean(((predict(net, xv, mu, sd) - gv) / sd) ** 2, axis=1)
                cands.append(dict(name=f'p{degree}r{r}{suffix}', degree=degree, rank=r,
                                  mse=float(err.mean()),
                                  se=float(err.std() / np.sqrt(len(err))), net=net))
    return cands


def select_candidate(cands, exact_tol=1e-3):
    """Screen for an (almost) exact polynomial, then apply the 1-SE rule.

    A polynomial expert is used only when its held-out unweighted MSE is below
    ``exact_tol``. Otherwise the neural fallback is returned. This prevents a
    polynomial that fits the mean response from being extrapolated into a
    rare-event region of a non-polynomial model.
    """
    poly = [c for c in cands if c['degree'] > 0 and c['mse'] <= exact_tol]
    if not poly:
        return min(cands, key=lambda c: (c['degree'] != 0, c['mse']))
    best = min(poly, key=lambda c: c['mse'])
    ok = [c for c in poly if c['mse'] <= best['mse'] + best['se']]
    return min(ok, key=lambda c: (c['degree'], c['rank']))
