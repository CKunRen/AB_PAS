"""Conditional-conflict adaptive deep subspaces: reproducible research prototype.

Reference inputs must be independent standard normals. No reference probability,
true latent basis or test labels are visible to the learner. Probe scores are
local diagnostics, not unbiased estimates of global conditional variance.
"""
import copy
import time
import numpy as np
import torch
from torch import nn
from reproducibility import seed_everything


class EventNet(nn.Module):
    def __init__(self, d, p, m, seed, basis=None, full=False):
        super().__init__()
        seed_everything(seed, True)
        self.d, self.m, self.full = d, m, full
        if not full:
            if basis is None:
                basis = np.random.default_rng(seed).normal(size=(d, m))
            self.raw = nn.Parameter(torch.tensor(basis, dtype=torch.float32))
        self.body = nn.Sequential(nn.Linear(m, 48), nn.Tanh(),
                                  nn.Linear(48, 48), nn.Tanh(), nn.Linear(48, p))

    def basis(self):
        if self.full:
            return torch.eye(self.d, device=next(self.parameters()).device)
        q, r = torch.linalg.qr(self.raw, mode='reduced')
        sign = torch.where(torch.diagonal(r) >= 0, 1., -1.)
        return q * sign

    def forward(self, u):
        return self.body(u if self.full else u @ self.basis())


def initial_basis(x, g, m, seed):
    # Response-supervised initialisation uses labels only, not latent truth.
    c = x.T @ (g-g.mean(0)) / len(x)
    rng = np.random.default_rng(seed)
    u, _, _ = np.linalg.svd(c, full_matrices=False)
    a = np.column_stack([u, rng.normal(size=(x.shape[1], m))])
    q, _ = np.linalg.qr(a)
    return q[:, :m]


def train(net, x, g, mu, sd, seed, epochs=200, weighted=True):
    seed_everything(seed, True)
    device = next(net.parameters()).device
    xt = torch.tensor(x, dtype=torch.float32, device=device)
    yt = torch.tensor((g-mu)/sd, dtype=torch.float32, device=device)
    near = np.min(np.abs(g/sd), axis=1)
    kap = np.exp(-.5*(near/.6)**2)
    kap /= max(kap.mean(), 1e-12)
    wt = torch.tensor(1.+5.*kap if weighted else np.ones(len(g)),
                      dtype=torch.float32, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=.003)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=.0001)
    net.train()
    for _ in range(epochs):
        pred = net(xt)
        loss = (wt*((pred-yt)**2).mean(1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
        opt.step()
        sched.step()
    return float(loss.detach().cpu())


def predict(net, x, mu, sd, chunk=16384):
    device = next(net.parameters()).device
    out = []
    net.eval()
    with torch.no_grad():
        for k in range(0, len(x), chunk):
            z = torch.tensor(x[k:k+chunk], dtype=torch.float32, device=device)
            out.append(net(z).cpu().numpy())
    return np.vstack(out)*sd+mu


def grow(net, x, residual, new_m, seed):
    old = net.basis().detach().cpu().numpy()
    c = x.T @ residual / len(x)
    c -= old @ (old.T @ c)
    u, _, _ = np.linalg.svd(c, full_matrices=False)
    a = np.column_stack([old, u, np.random.default_rng(seed).normal(size=(net.d,new_m))])
    q, _ = np.linalg.qr(a)
    q[:, :net.m] *= np.where(np.sum(q[:, :net.m]*old,axis=0)>=0,1.,-1.)
    new = EventNet(net.d, residual.shape[1], new_m, seed, q[:, :new_m])
    new.to(next(net.parameters()).device)
    state = copy.deepcopy(net.body.state_dict())
    with torch.no_grad():
        new.body[0].weight[:, :net.m] = net.body[0].weight
        new.body[0].weight[:, net.m:] *= .1
        for key, val in state.items():
            if key != '0.weight':
                new.body.state_dict()[key].copy_(val)
    return new


def wilson_lower(k, n):
    z=1.96; ph=k/n
    return (ph+z*z/(2*n)-z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n)))/(1+z*z/n)


def run(problem, method, seed, epochs=200, initial=600, increment=150,
        rounds=4, pool_size=12000, device='cpu', evaluator=None,
        return_state=False):
    """All labelled points, including validation and probes, count once.

    static: labelled holdout selects rank 2/4/8 from a passive full-budget design.
    active8: boundary AL, rank fixed at 8.
    conflict: local conditional conflict controls nested rank expansion.
    fixed2: same probes/acquisition, but disables rank expansion.
    full: passive full-input neural network.
    evaluator is a callback receiving a predictor after learning, never labels.
    """
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    n_calls = 0
    def call(x):
        nonlocal n_calls
        n_calls += len(x)
        return problem.evaluate(x)
    x0 = rng.normal(size=(initial, problem.d))
    g0 = call(x0)
    nv=120
    xv, gv = x0[:nv], g0[:nv]
    x, g = x0[nv:], g0[nv:]
    mu, sd = g.mean(0), g.std(0)+1e-8
    logs=[]
    def score(net):
        gh=predict(net,xv,mu,sd)
        k=np.exp(-.5*(np.min(np.abs(gv/sd),axis=1)/.6)**2)
        k/=max(k.mean(),1e-12)
        return float(np.mean(k*np.mean(((gh-gv)/sd)**2,axis=1)))
    if method in ('static','full'):
        xn=rng.normal(size=(increment*rounds,problem.d)); gn=call(xn)
        x=np.vstack([x,xn]); g=np.vstack([g,gn])
        candidates=[problem.d] if method=='full' else [2,4,8]
        candidates=sorted(set(min(m,problem.d) for m in candidates))
        fits=[]
        for m in candidates:
            net=EventNet(problem.d,problem.p,m,seed+100,initial_basis(x,g,m,seed),method=='full').to(device)
            train(net,x,g,mu,sd,seed+100,epochs*2)
            fits.append((score(net),m,net))
        best=min(v[0] for v in fits)
        _, m, net=min((v for v in fits if v[0]<=1.05*best),key=lambda v:v[1])
        logs.append(dict(calls=n_calls,rank=m,holdout=score(net)))
    else:
        m=min(8 if method=='active8' else 2,problem.d)
        net=EventNet(problem.d,problem.p,m,seed+100,initial_basis(x,g,m,seed)).to(device)
        train(net,x,g,mu,sd,seed+100,epochs)
        logs.append(dict(calls=n_calls,rank=m,holdout=score(net)))
        for t in range(rounds):
            xc=rng.normal(size=(pool_size,problem.d))
            gh=predict(net,xc,mu,sd)
            # Per-mode boundary coverage plus global exploration.
            margin=np.abs(gh/sd)
            ranks=[np.argsort(margin[:,j])[:increment] for j in range(problem.p)]
            order=[int(ranks[j][k]) for k in range(increment) for j in range(problem.p)]
            order=list(dict.fromkeys(order))
            probe_n=32 if method in ('conflict','fixed2') and m<problem.d else 0
            conflicts=0; local_rate=None; modal_rate=None; lower=None
            if probe_n:
                w=net.basis().detach().cpu().numpy()
                # Half boundary anchors, half random anchors; deliberately local.
                anchors=np.vstack([xc[order[:probe_n//2]],rng.normal(size=(probe_n-probe_n//2,problem.d))])
                base=(anchors@w)@w.T
                v=rng.normal(size=(probe_n,problem.d))
                vp=rng.normal(size=(probe_n,problem.d))
                xp=base+v-(v@w)@w.T
                xpp=base+vp-(vp@w)@w.T
                gp=call(xp); gpp=call(xpp)
                y=problem.event(gp); yp=problem.event(gpp)
                conflicts=int(np.count_nonzero(y!=yp))
                modal_rate=np.mean((gp<=0)!=(gpp<=0),axis=0).tolist()
                local_rate=conflicts/probe_n
                lower=wilson_lower(conflicts,probe_n)
                xn=np.vstack([xp,xpp]); gn=np.vstack([gp,gpp])
                if method=='conflict' and lower>.02 and m<min(problem.d,12):
                    residual=(gn-predict(net,xn,mu,sd))/sd
                    net=grow(net,xn,residual,min(m+2,problem.d,12),seed+200+t)
                    m=net.m
            else:
                xn=np.empty((0,problem.d)); gn=np.empty((0,problem.p))
            remaining=increment-len(xn)
            nr=remaining//2
            xb=np.vstack([xc[order[:remaining-nr]],rng.normal(size=(nr,problem.d))])
            gb=call(xb)
            x=np.vstack([x,xn,xb]); g=np.vstack([g,gn,gb])
            train(net,x,g,mu,sd,seed+300+t,epochs)
            logs.append(dict(calls=n_calls,rank=m,holdout=score(net),
                             probe_pairs=probe_n,conflicts=conflicts,local_conflict=local_rate,
                             mode_conflict=modal_rate,wilson_lower=lower))
    assert n_calls==initial+increment*rounds
    train_seconds=time.perf_counter()-start
    pred=lambda a: predict(net,a,mu,sd)
    result=dict(method=method,seed=seed,calls=n_calls,rank=net.m,history=logs,
                training_seconds=train_seconds,validation_count=nv,
                probe_calls=sum(2*v.get('probe_pairs',0) for v in logs))
    if evaluator:
        te=time.perf_counter(); result.update(evaluator(pred))
        result['inference_seconds']=time.perf_counter()-te
    if return_state:
        result['state']=dict(net=net,x=x,g=g,xv=xv,gv=gv,mu=mu,sd=sd,
                             basis=net.basis().detach().cpu().numpy())
    result['seconds']=time.perf_counter()-start
    return result
