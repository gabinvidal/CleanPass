import numpy as np, firefly, sys
from scipy.ndimage import median_filter, maximum_filter
import os
d = os.environ.get('DATA', 'data') + '/'
B = np.load(d+'beauty_f1.npy'); R = np.load(d+'reference.npy')
ALB = np.load(d+'albedo.npy'); NRM = np.load(d+'normal.npy')
P0 = np.load(d+'beauty_f0.npy'); P2 = np.load(d+'beauty_f2.npy')
E = lambda x: np.clip(x, 0, None).max(-1)
ER = E(R)
# legit features: isolated highlights in the converged reference
legit = ER > 4 * median_filter(ER, size=5)
def metrics(out, name):
    eo, eb = E(out), E(B)
    relmse = np.mean(((out - R) ** 2 / (R ** 2 + 1e-2)))
    gross = (eo > 8 * ER + 0.5).sum()
    changed = (np.abs(out - B).max(-1) > 1e-6).mean() * 100
    # legit energy kept (ratio of output vs input at legit pixels)
    keep = eo[legit].sum() / max(eb[legit].sum(), 1e-6)
    print(f'{name:28s} relMSE {relmse:9.4f}  gross {gross:5d}  changed {changed:6.2f}%  legit kept {keep:5.3f}')
if __name__ == '__main__':
    print('legit pixels', legit.sum(), ' input gross outliers', (E(B) > 8*ER + 0.5).sum())
    metrics(B, 'input')
    # naive baseline: clamp to 4x 3x3 median
    m3 = median_filter(B, size=(3, 3, 1))
    metrics(np.minimum(B, 4 * m3 + 1e-3), 'baseline clamp 4x median3')
    for ref in (1, 0):
        for meth in (0, 1, 2):
            out, _, matte, st = firefly.run(B, ALB, NRM, reference=ref, method=meth)
            metrics(out, f'firefly ref={ref} meth={meth}')
        print('   hits core/total', st[0], st[1], 'time %.2fs' % st[2])
    out, _, matte, st = firefly.run(B, ALB, NRM, P0, P2, reference=0, temporal=1)
    metrics(out, 'firefly oidn + temporal')
    out, _, matte, st = firefly.run(B, None, None, reference=0)
    metrics(out, 'firefly oidn no guides')
    out, _, matte, st = firefly.run(B, None, None, reference=1)
    metrics(out, 'firefly stat no guides')
