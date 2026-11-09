"""Hard content for the temporal chain: the seq2 shot (camera pan, moving ball)
with fog (volume), hair (sub-pixel curves), depth of field or motion blur.
Writes, per variant v and frame f, into $OUT (default ./d3):
  {v}_beauty_f, {v}_var_f      noisy beauty (4 x 8 spp, animated seed) and its variance
  {v}_ref_f                    reference (REFSPP, default 1024)
  {v}_alb_f, {v}_nrm_f         guides (32 spp, same lens / shutter as the beauty)
  {v}_z_f, {v}_mv_f            depth and forward motion vectors (px, top-down, full frame)
Usage: python render_seq3.py vol|hair|dof|mblur [ref]
"""
import os, sys, time, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv = sys.argv
sys.argv = ['x', 'none']
import render_seq2 as r
sys.argv = _argv
import mitsuba as mi

OUT = os.environ.get('OUT', os.path.join(HERE, 'd3'))
os.makedirs(OUT, exist_ok=True)
NF = r.NF
RES = r.RES
FOV = 39.3077
REFSPP = int(os.environ.get('REFSPP', 1024))


def ball(v, f):
    """ball centre at (possibly fractional) frame f"""
    if v == 'mblur':  # fast: ~12 px / frame
        return [-0.75 + 0.15 * f, 0.15, 0.25]
    return r.ball(f)


def hair_file():
    p = os.path.join(OUT, 'hair.txt')
    if not os.path.exists(p):
        rng = np.random.default_rng(1)
        L = []
        for i in range(1500):  # a curtain of strands hanging from the ceiling, behind the ball
            x, z = rng.uniform(-0.75, 0.75), rng.uniform(-0.7, -0.1)
            ph = rng.uniform(0, 6.28)
            n = 8
            for k in range(n):
                y = 0.99 - 0.11 * k
                L.append('%f %f %f %f' % (x + 0.04 * np.sin(ph + 0.7 * k), y, z + 0.03 * np.cos(ph + 0.5 * k), 0.0018))
            L.append('')
        open(p, 'w').write('\n'.join(L))
    return p


def scene(v, f, integ=None, pinhole=False):
    """f may be fractional (motion blur sub-frames: only the ball moves within the shutter)"""
    fc = int(round(f)) if v == 'mblur' else f
    sc = r.scene(fc, 'AB', integ)
    sc['ball']['center'] = ball(v, f)
    if v == 'vol' and not (integ and integ.get('type') == 'aov'):  # guides / vectors: the surfaces behind the fog
        sc['fog'] = {'type': 'cube', 'to_world': mi.ScalarTransform4f().scale([0.985, 0.985, 0.985]), 'bsdf': {'type': 'null'},
                     'interior': {'type': 'homogeneous', 'sigma_t': 0.6, 'albedo': 0.85}}
        if not integ:
            sc['integrator'] = {'type': 'volpath', 'max_depth': 10}
    if v == 'hair':
        sc['hair'] = {'type': 'linearcurve', 'filename': hair_file(),
                      'bsdf': {'type': 'roughplastic', 'diffuse_reflectance': {'type': 'rgb', 'value': [0.35, 0.2, 0.08]}, 'alpha': 0.25}}
    if v == 'dof' and not pinhole:
        o, t = r.cam(fc)
        s = sc['sensor']
        sc['sensor'] = {'type': 'thinlens', 'fov': s.get('fov', FOV), 'to_world': s['to_world'], 'film': s['film'],
                        'aperture_radius': 0.16, 'focus_distance': 3.65}
        if 'sampler' in s:
            sc['sensor']['sampler'] = s['sampler']
    return sc


def render(sc, spp, seed):
    return np.array(mi.render(mi.load_dict(sc), spp=spp, seed=seed), np.float32)


def shutter(v, f, spp, seed, integ=None):
    """beauty or guides over the shutter (8 sub-frames for mblur, centred, 0.5 frame open)"""
    if v != 'mblur':
        return render(scene(v, f, integ), spp, seed)
    subs = [render(scene(v, f + 0.5 * ((k + 0.5) / 8 - 0.5), integ), max(1, spp // 8), seed + 97 * k) for k in range(8)]
    return np.mean(subs, 0)


def proj(P, f):
    o, t = r.cam(f)
    o = np.array(o); t = np.array(t); up = np.array([0, 1., 0])
    d = (t - o) / np.linalg.norm(t - o); left = np.cross(up, d); left /= np.linalg.norm(left); nu = np.cross(d, left)
    w = P - o; zc = w @ d; xc = w @ left; yc = w @ nu
    th = np.tan(np.radians(FOV) / 2)
    return RES / 2 - RES / 2 * xc / zc / th, RES / 2 - RES / 2 * yc / zc / th, zc


def vectors(v, f):
    g = render(scene(v, f, {'type': 'aov', 'aovs': 'pos:position,sid:shape_index'}, pinhole=True), 1, 3)
    pos = g[..., 0:3].reshape(-1, 3); sid = g[..., 3].ravel()
    x, y, _ = proj(np.array(ball(v, f)), f)
    bs = sid[int(np.clip(y, 0, RES - 1)) * RES + int(np.clip(x, 0, RES - 1))]
    mb = np.abs(sid - bs) < 0.5
    P = pos.copy()
    P[mb] += np.array(ball(v, f + 1)) - np.array(ball(v, f))
    x0, y0, z = proj(pos, f)
    x1, y1, _ = proj(P, f + 1)
    return np.stack([x1 - x0, y1 - y0], -1).reshape(RES, RES, 2).astype(np.float32), z.reshape(RES, RES).astype(np.float32)


if __name__ == '__main__':
    v = sys.argv[1]
    what = sys.argv[2] if len(sys.argv) > 2 else 'noisy'
    mi.set_variant('llvm_ad_rgb')
    for f in range(NF):
        t0 = time.time()
        if what == 'noisy':
            subs = np.stack([shutter(v, f, 8, 1000 + 10 * k + 100 * f)[..., :3] for k in range(4)])
            np.save(f'{OUT}/{v}_beauty_{f}.npy', subs.mean(0))
            np.save(f'{OUT}/{v}_var_{f}.npy', subs.var(0, ddof=1) / 4)
            g = shutter(v, f, 32, 7, {'type': 'aov', 'aovs': 'alb:albedo,nn:sh_normal'})
            np.save(f'{OUT}/{v}_alb_{f}.npy', g[..., 0:3]); np.save(f'{OUT}/{v}_nrm_{f}.npy', g[..., 3:6])
            mv, z = vectors(v, f)
            np.save(f'{OUT}/{v}_mv_{f}.npy', mv); np.save(f'{OUT}/{v}_z_{f}.npy', z)
        elif what == 'ref':
            np.save(f'{OUT}/{v}_ref_{f}.npy', shutter(v, f, REFSPP, 50000 + 7 * f)[..., :3])
        print(v, what, f, round(time.time() - t0, 1), flush=True)
