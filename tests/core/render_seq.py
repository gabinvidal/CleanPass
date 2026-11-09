import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
import render_scenes as rs
rs.RES = 320
def sd(frame):
    sc = rs.scene_dict(1, 'AB')
    dx = (frame - 3) * 1.3 * (2.0 / rs.RES) * 0.9
    sc['sensor']['to_world'] = mi.ScalarTransform4f().look_at(origin=[dx, 0.02 * (frame - 3) * 0.3, 3.9], target=[dx, 0, 0], up=[0, 1, 0])
    sc['sensor']['film']['width'] = sc['sensor']['film']['height'] = rs.RES
    # finer checker -> aliasing / crawling texture
    sc['floor']['bsdf']['reflectance']['to_uv'] = mi.ScalarTransform4f().scale([22, 22, 1])
    return sc
what = sys.argv[1]
for f in range(7):
    sc = sd(f)
    if what == 'noisy':
        img = np.array(mi.render(mi.load_dict(sc), spp=48, seed=100 + f), np.float32)[..., :3]
        np.save(f'noisy_{f}.npy', img)
        sc2 = dict(sc); sc2['integrator'] = {'type': 'aov', 'aovs': 'alb:albedo,nn:sh_normal'}
        g = np.array(mi.render(mi.load_dict(sc2), spp=32, seed=7), np.float32)
        np.save(f'alb_{f}.npy', g[..., :3]); np.save(f'nrm_{f}.npy', g[..., 3:6])
    else:
        img = np.array(mi.render(mi.load_dict(sc), spp=1024, seed=900 + f), np.float32)[..., :3]
        np.save(f'ref_{f}.npy', img)
    print(what, f, flush=True)
