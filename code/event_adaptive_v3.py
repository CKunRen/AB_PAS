"""CC-DAS v3 prototype: validation-gated rank expansion.

This module does not modify any frozen file. It reuses the frozen encoder,
response network, probe construction and residual-direction proposal, but
replaces the "any conflict -> expand" gate by a held-out comparison of the
current rank and a proposed rank m+2. Both candidates are trained on exactly
the same labelled data; the larger rank is accepted only if its holdout score
improves by a fixed margin. The conflict probe is retained as the trigger
(``gate='conflict_validation'``) or as a diagnostic only
(``gate='validation'``).
"""
import copy
import time

import numpy as np
import torch

from event_adaptive_counterexample import (
    EventNet, initial_basis, train, predict, grow, wilson_lower,
)
from reproducibility import seed_everything


def run(problem, method, seed, epochs=200, initial=600, increment=150,
        rounds=4, pool_size=12000, device='cpu', evaluator=None,
        gate='conflict_validation', gain=0.02, candidate_epochs=None,
        return_state=False, use_probes=True):
    """Run one trajectory with a validation-gated expansion rule.

    gate:
      'conflict_validation' : propose m+2 only when a true conflict is observed
      'validation'          : always compare m and m+2
      'counterexample'      : frozen baseline (any conflict -> grow)
    gain: relative holdout improvement required to accept m+2.
    """
    if candidate_epochs is None:
        candidate_epochs = epochs
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    n_calls = 0

    def call(x):
        nonlocal n_calls
        n_calls += len(x)
        return problem.evaluate(x)

    x0 = rng.normal(size=(initial, problem.d))
    g0 = call(x0)
    nv = 120
    xv, gv = x0[:nv], g0[:nv]
    x, g = x0[nv:], g0[nv:]
    mu, sd = g.mean(0), g.std(0) + 1e-8

    def score(net):
        gh = predict(net, xv, mu, sd)
        k = np.exp(-0.5 * (np.min(np.abs(gv / sd), axis=1) / 0.6) ** 2)
        k /= max(k.mean(), 1e-12)
        return float(np.mean(k * np.mean(((gh - gv) / sd) ** 2, axis=1)))

    logs = []
    m = min(8 if method == 'active8' else 2, problem.d)
    net = EventNet(problem.d, problem.p, m, seed + 100,
                   initial_basis(x, g, m, seed)).to(device)
    train(net, x, g, mu, sd, seed + 100, epochs)
    logs.append(dict(calls=n_calls, rank=m, holdout=score(net), grown=False))

    for t in range(rounds):
        xc = rng.normal(size=(pool_size, problem.d))
        gh = predict(net, xc, mu, sd)
        margin = np.abs(gh / sd)
        ranks = [np.argsort(margin[:, j])[:increment] for j in range(problem.p)]
        order = [int(ranks[j][k]) for k in range(increment) for j in range(problem.p)]
        order = list(dict.fromkeys(order))

        probe_n = 32 if use_probes and m < problem.d else 0
        conflicts = 0
        local_rate = None
        lower = None
        if probe_n:
            w = net.basis().detach().cpu().numpy()
            anchors = np.vstack([xc[order[:probe_n // 2]],
                                 rng.normal(size=(probe_n - probe_n // 2, problem.d))])
            base = (anchors @ w) @ w.T
            v = rng.normal(size=(probe_n, problem.d))
            vp = rng.normal(size=(probe_n, problem.d))
            xp = base + v - (v @ w) @ w.T
            xpp = base + vp - (vp @ w) @ w.T
            gp = call(xp)
            gpp = call(xpp)
            y = problem.event(gp)
            yp = problem.event(gpp)
            conflicts = int(np.count_nonzero(y != yp))
            local_rate = conflicts / probe_n
            lower = wilson_lower(conflicts, probe_n)
            xn = np.vstack([xp, xpp])
            gn = np.vstack([gp, gpp])
        else:
            xn = np.empty((0, problem.d))
            gn = np.empty((0, problem.p))

        remaining = increment - len(xn)
        nr = remaining // 2
        xb = np.vstack([xc[order[:remaining - nr]],
                        rng.normal(size=(nr, problem.d))])
        gb = call(xb)
        x_aug = np.vstack([x, xn, xb])
        g_aug = np.vstack([g, gn, gb])

        # Current-rank candidate: continue training from the existing net.
        cur = copy.deepcopy(net)
        train(cur, x_aug, g_aug, mu, sd, seed + 300 + t, epochs)
        cur_score = score(cur)

        grown = False
        chosen = cur
        cand_score = None
        if m < min(problem.d, 12) and (gate == 'validation' or conflicts > 0):
            # Direction proposal uses the pre-growth net and the probe residuals.
            if probe_n:
                residual = (gn - predict(net, xn, mu, sd)) / sd
                grow_x = xn
            else:
                residual = (g_aug - predict(net, x_aug, mu, sd)) / sd
                grow_x = x_aug
            cand = grow(net, grow_x, residual, min(m + 2, problem.d, 12),
                        seed + 200 + t)
            train(cand, x_aug, g_aug, mu, sd, seed + 400 + t, candidate_epochs)
            cand_score = score(cand)
            if cand_score < cur_score * (1.0 - gain):
                chosen = cand
                grown = True

        net = chosen
        m = net.m
        x, g = x_aug, g_aug
        logs.append(dict(calls=n_calls, rank=m, holdout=score(net),
                         probe_pairs=probe_n, conflicts=conflicts,
                         local_conflict=local_rate, wilson_lower=lower,
                         current_score=cur_score, candidate_score=cand_score,
                         grown=grown))

    assert n_calls == initial + increment * rounds
    train_seconds = time.perf_counter() - start
    pred = lambda a: predict(net, a, mu, sd)
    result = dict(method=method, seed=seed, calls=n_calls, rank=net.m,
                  history=logs, training_seconds=train_seconds,
                  validation_count=nv,
                  probe_calls=sum(2 * v.get('probe_pairs', 0) for v in logs))
    if return_state:
        result['state'] = dict(net=net, 
            basis=net.basis().detach().cpu().numpy(),
            x=x, g=g, mu=mu, sd=sd, xv=xv, gv=gv,
            gh_v=predict(net, xv, mu, sd),
        )
    if evaluator:
        te = time.perf_counter()
        result.update(evaluator(pred))
        result['inference_seconds'] = time.perf_counter() - te
    result['seconds'] = time.perf_counter() - start
    return result
