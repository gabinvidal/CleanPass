"""Exact-ground-truth test: inject fireflies (single samples seen through a
Gaussian pixel filter) into the converged render and its light-group AOVs."""
import numpy as np, firefly, sys
from evaluate import E, legit
d = 'data/'
RA, RB = np.load(d + 'refA.npy'), np.load(d + 'refB.npy')
ALB, NRM = np.load(d + 'albedo.npy'), np.load(d + 'normal.npy')
H, W, _ = RA.shape
rng = np.random.default_rng(3)

def inject(n=500, sigma=0.5, seed=3):
    rng = np.random.default_rng(seed)
    A, Bb = RA.copy(), RB.copy()
    gt = np.zeros((H, W), np.float32)  # injected energy
    base = E(RA + RB)
    loc = []
    for _ in range(n):
        y, x = rng.uniform(3, H - 3), rng.uniform(3, W - 3)
        iy, ix = int(y), int(x)
        lvl = max(base[iy, ix], 0.05)
        amp = lvl * 2 ** rng.uniform(2.5, 10)
        col = rng.uniform(0.3, 1.0, 3); col /= col.max()
        tgt = A if rng.random() < 0.5 else Bb
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                py, px = iy + dy, ix + dx
                w = np.exp(-((py + .5 - y) ** 2 + (px + .5 - x) ** 2) / (2 * sigma ** 2))
                tgt[py, px] += amp * col * w / (2 * np.pi * sigma ** 2)
                gt[py, px] += amp * w / (2 * np.pi * sigma ** 2)
        loc.append((iy, ix))
    return A, Bb, gt, loc

if __name__ == '__main__':
    A, Bb, gt, loc = inject()
    Bt = A + Bb; Rt = RA + RB
    kw = dict(threshold=1.5, isolation=0.75, adaptivity=0.25, grow=1)
    kw.update({k: float(v) for k, v in (a.split('=') for a in sys.argv[1:])})
    inj = gt > 0.02 * E(Rt)  # significantly contaminated pixels
    for ref in (0, 1):
        out, aovs, m, st = firefly.run(Bt, ALB, NRM, aovs=[A, Bb], reference=ref, **kw)
        removed_inj = np.clip(E(Bt) - E(out), 0, None)[inj].sum() / gt[inj].sum()
        err_in = np.abs(Bt - Rt)[inj].sum(); err_out = np.abs(out - Rt)[inj].sum()
        fp = np.abs(out - Bt)[~inj].sum() / np.abs(Bt)[~inj].sum() * 100
        fp_px = (np.abs(out - Bt).max(-1) > 1e-6)[~inj].sum()
        left = 0
        for (y, x) in loc:
            if E(out)[y, x] > 2 * E(Rt)[y, x] + 0.05: left += 1
        aov_sum_err = np.abs(aovs[0] + aovs[1] - out).max()
        print(f'ref={ref}: removed {removed_inj*100:5.1f}% of injected energy | error on contaminated px {err_in:9.1f} -> {err_out:8.1f} | '
              f'fireflies left {left}/{len(loc)} | untouched-area change {fp:.4f}% ({fp_px} px) | legit lost {(E(Bt)-E(out))[legit & ~inj].sum():.2f} | AOV sum err {aov_sum_err:.2e} | {st[2]:.2f}s')
