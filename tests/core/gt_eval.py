import os
import numpy as np, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pipe, view
N = {f: np.load(f'noisy_{f}.npy') for f in range(7)}
R = {f: np.load(f'ref_{f}.npy') for f in range(7)}
A = {f: np.load(f'alb_{f}.npy') for f in range(7)}
Nn = {f: np.load(f'nrm_{f}.npy') for f in range(7)}
def run(flist, **kw):
    R2 = max(1, kw.get('frames', 1))
    out = {}
    for f in flist:
        offs = [o for o in (-1, 1, -2, 2) if 0 <= f + o < 7 and (abs(o) <= R2)]
        o, m, st = pipe.run(N[f], [N[f + o] for o in offs], offs, A[f], Nn[f], [A[f + o] for o in offs], [Nn[f + o] for o in offs], **kw)
        out[f] = o
    return out
def tm(x): x = np.clip(x, 0, None); return x / (1 + x)   # perceptual-ish (display) domain
def score(out, frames):
    err = np.mean([np.abs(tm(out[f]) - tm(R[f])).mean() for f in frames])
    fl = np.mean([np.abs((tm(out[f]) - tm(R[f])) - (tm(out[f - 1]) - tm(R[f - 1]))).mean() for f in frames[1:]])
    return err, fl
frames = [2, 3, 4]
base = {f: N[f] for f in frames}
e, fl = score(base, frames); print('%-60s err %.5f flicker %.5f' % ('input', e, fl))
for a in sys.argv[1:]:
    kw = json.loads(a)
    out = run(frames, **kw)
    e, fl = score(out, frames)
    print('%-60s err %.5f flicker %.5f' % (a, e, fl), flush=True)
