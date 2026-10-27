"""v6: simplified main method.

Fixed rank-2 MLP boundary sampler (no conditional probes, no growth) +
parsimonious polynomial experts (degree 2/3, ranks 2/5/8/20, warm and random
initialisation, unweighted MSE, 1-SE selection) + MLP fallback.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtri
from scipy.stats import qmc

from event_adaptive_v3 import run
from method_benchmarks import Problem
from poly_experts import fit_candidates, select_candidate, predict

ROOT = Path(__file__).resolve().parent
GRID = {2: [2, 5, 8, 20], 3: [2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--cases', default='1,2,3,4,5,6,7')
    ap.add_argument('--power', type=int, default=18)
    ap.add_argument('--output', default='results/method_paper_v3/v6_final_reduced.json')
    args = ap.parse_args()
    torch.set_num_threads(2)
    dest = ROOT / args.output
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(dest.read_text(encoding='utf-8')) if dest.exists() else {'cases': {}}
    for n in map(int, args.cases.split(',')):
        p = Problem(n)
        ref, _ = p.reference()
        ev = ndtri(np.clip(qmc.Sobol(p.d, scramble=True, seed=72000 + n)
                           .random_base2(args.power), 1e-12, 1 - 1e-12))
        def evaluate(pred):
            return dict(Pf=float(p.event(pred(ev)).mean()))
        c = data['cases'].setdefault(str(n), {'name': p.name, 'reference': ref,
                                              'runs': []})
        for s in range(args.seeds):
            if any(r['seed_index'] == s for r in c['runs']):
                continue
            t0 = time.time()
            row = run(p, 'conflict', 410000 + n * 1000 + s, epochs=300,
                      gate='conflict_validation', gain=0.02, return_state=True,
                      evaluator=evaluate, use_probes=False)
            st = row['state']
            gv = st['gv']; sd = st['sd']
            err_mlp = np.mean(((st['gh_v'] - gv) / sd) ** 2, axis=1)
            mlp_mse = float(err_mlp.mean())
            cands = fit_candidates(st['x'], st['g'], st['xv'], gv,
                                   st['mu'], sd, st['basis'], p,
                                   mlp_mse, 410000 + n * 1000 + s + 7000,
                                   grid=GRID, epochs=1500,
                                   random_ranks={2, 20})
            for cand in cands:
                if cand['name'] == 'mlp':
                    cand['se'] = float(err_mlp.std() / np.sqrt(len(err_mlp)))
            choice = select_candidate(cands)
            pf = row['Pf'] if choice['name'] == 'mlp' else float(
                p.event(predict(choice['net'], ev, st['mu'], sd)).mean())
            c['runs'].append(dict(seed_index=s, rank=row['rank'], mlp=row['Pf'],
                                  selected=pf, choice=choice['name'],
                                  mlp_mse=mlp_mse, choice_mse=choice['mse'],
                                  best_mse=min(x['mse'] for x in cands),
                                  seconds=time.time() - t0))
            dest.write_text(json.dumps(data, indent=2), encoding='utf-8')
            print(f'case {n} seed {s} rank {row["rank"]} mlp={row["Pf"]:.6g} '
                  f'choice={choice["name"]}:{pf:.6g} t={time.time()-t0:.1f}s',
                  flush=True)
        arr = np.array([[r['mlp'], r['selected']] for r in c['runs']])
        for k, name in enumerate(['mlp', 'selected']):
            rel = np.sqrt(np.mean((arr[:, k] - ref) ** 2)) / ref
            print(f'== case {n} {name:8s} relRMSE={rel:.4f} '
                  f'bias={np.mean(arr[:, k] - ref) / ref:+.3f}', flush=True)
        print('choices', [r['choice'] for r in c['runs']], flush=True)


if __name__ == '__main__':
    main()
