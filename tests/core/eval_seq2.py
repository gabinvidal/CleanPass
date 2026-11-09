import sys, json, numpy as np
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pipe4
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'd') + '/'
NF = 9
L = lambda n: np.load(D + n + '.npy')
def data(mode='anim'):
    F = {}
    for f in range(NF):
        F[f] = dict(beauty=L(f'{mode}A_{f}') + L(f'{mode}B_{f}'), A=L(f'{mode}A_{f}'), B=L(f'{mode}B_{f}'),
                    variance=L(f'{mode}Avar_{f}') + L(f'{mode}Bvar_{f}'), albedo=L(f'alb_{f}'), normal=L(f'nrm_{f}'),
                    depth=L(f'depth_{f}'), ref=L(f'refA_{f}') + L(f'refB_{f}'), refA=L(f'refA_{f}'), refB=L(f'refB_{f}'),
                    ball=L(f'ballmask_{f}'))
        F[f]['mvf'] = L(f'mvf_{f}') if f + 1 < NF else -L(f'mvb_{f}')
    return F
def conv(mv, how, W=256):
    if how == 'fwd': return mv
    if how == 'bwd': return -mv
    if how == 'yflip': return mv * np.array([1, -1], np.float32)
    if how == 'ndc': return mv / (W / 2)
    if how == 'half': return mv * 0.5
    if how == 'noise': return mv + np.random.default_rng(0).normal(0, 3, mv.shape).astype(np.float32)
    raise ValueError(how)
def run(F, t, frames=1, mv=None, use=('albedo', 'normal'), nbr=None, **kw):
    nbr = nbr or (2 if kw.get('stabilize') else frames)
    keys = ['beauty'] + list(use)
    cur = {k: F[t][k] for k in keys}
    if mv: cur['motion'] = conv(F[t]['mvf'], mv)
    nbs = []
    for o in (-1, 1, -2, 2):
        if abs(o) <= nbr and 0 <= t + o < NF:
            d = {k: F[t + o][k] for k in keys}; d['offset'] = o
            if mv: d['motion'] = conv(F[t + o]['mvf'], mv)
            nbs.append(d)
    return pipe4.run(cur, nbs, frames=frames, **kw)
def tm(x): x = np.clip(x, 0, None); return x / (1 + x)
def score(F, out, frames, mask=None):
    m = (lambda a: a[mask[0]]) if mask is not None else (lambda a: a)
    err = np.mean([np.abs(tm(out[f]) - tm(F[f]['ref'])).mean(-1)[F[f]['ball'] if mask == 'ball' else slice(None)].mean() for f in frames])
    fl = np.mean([np.abs((tm(out[f]) - tm(F[f]['ref'])) - (tm(out[f - 1]) - tm(F[f - 1]['ref']))).mean() for f in frames[1:]])
    return err, fl
