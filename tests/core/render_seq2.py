"""Moving-camera + moving-object sequence with exact motion vectors, depth,
per-pixel variance, two light groups, animated and fixed seed, and a
high-spp reference per frame (Mitsuba 3)."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
import render_scenes as rs

RES = 256
NF = 9
rs.RES = RES

def cam(f):
    dx = (f - 4) * 0.022          # ~1.5 px / frame pan
    dy = (f - 4) * 0.004
    return [dx, dy, 3.9], [dx * 0.6, 0, 0]

def ball(f):                     # moving diffuse ball, ~3 px / frame
    return [-0.55 + 0.045 * f, 0.15, 0.25]

def scene(f, light, integ=None, filt='box'):
    sc = rs.scene_dict(1, light)
    o, t = cam(f)
    sc['sensor']['to_world'] = mi.ScalarTransform4f().look_at(origin=o, target=t, up=[0, 1, 0])
    sc['sensor']['film']['width'] = sc['sensor']['film']['height'] = RES
    sc['sensor']['film']['rfilter'] = {'type': filt}
    sc['floor']['bsdf']['reflectance']['to_uv'] = mi.ScalarTransform4f().scale([14, 14, 1])
    sc['ball'] = {'type': 'sphere', 'center': ball(f), 'radius': 0.16,
                  'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'checkerboard', 'color0': 0.1, 'color1': 0.6,
                                                               'to_uv': mi.ScalarTransform4f().scale([6, 3, 1])}}}
    if integ:
        sc['integrator'] = integ
    return sc

def render(sc, spp, seed):
    return np.array(mi.render(mi.load_dict(sc), spp=spp, seed=seed), np.float32)

if __name__ == '__main__':
    what = sys.argv[1]
    os.makedirs('d', exist_ok=True)
    for f in range(NF):
        t0 = time.time()
        if what == 'aux':
            g = render(scene(f, 'AB', {'type': 'aov', 'aovs': 'alb:albedo,nn:sh_normal,pos:position,sid:shape_index,dd:depth'}), 16, 7)
            np.save(f'd/alb_{f}.npy', g[..., 0:3]); np.save(f'd/nrm_{f}.npy', g[..., 3:6])
            # 1 spp centre sample for position / id (no filter blending)
            g1 = render(scene(f, 'AB', {'type': 'aov', 'aovs': 'pos:position,sid:shape_index,dd:depth'}, 'box'), 1, 3)
            np.save(f'd/pos_{f}.npy', g1[..., 0:3]); np.save(f'd/sid_{f}.npy', g1[..., 3]); np.save(f'd/z_{f}.npy', g1[..., 4])
        elif what in ('anim', 'fixed'):
            for L in 'AB':
                subs = []
                for k in range(4):
                    seed = (1000 if L == 'A' else 2000) + 10 * k + (f * 100 if what == 'anim' else 0)
                    subs.append(render(scene(f, L), 8, seed)[..., :3])
                subs = np.stack(subs)
                np.save(f'd/{what}{L}_{f}.npy', subs.mean(0))
                np.save(f'd/{what}{L}var_{f}.npy', subs.var(0, ddof=1) / 4)
        elif what == 'ref':
            for L in 'AB':
                np.save(f'd/ref{L}_{f}.npy', render(scene(f, L), int(os.environ.get('REFSPP', 768)), 50000 + f * 7 + (L == 'B'))[..., :3])
        print(what, f, round(time.time() - t0, 1), flush=True)
