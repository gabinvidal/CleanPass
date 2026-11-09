"""Render test material for the CleanPass core with Mitsuba 3.

Produces, in ./data/, float32 .npy images (H, W, 3):
  beauty_fN.npy      noisy beauty at frame N (N = 0,1,2), = lightA + lightB
  lightA_fN.npy      light group A (ceiling light) at frame N
  lightB_fN.npy      light group B (small side light) at frame N
  albedo.npy, normal.npy   clean-ish guides (frame 1)
  reference.npy      high-spp beauty of frame 1 (ground truth)
The camera moves 1 px/frame horizontally so the temporal check sees motion.
"""
import os, sys
import numpy as np
import mitsuba as mi

mi.set_variant('llvm_ad_rgb')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get('OUTDIR', 'data'))
os.makedirs(OUT, exist_ok=True)

RES = int(os.environ.get('RES', 384))
SPP = int(os.environ.get('SPP', 24))
REF_SPP = int(os.environ.get('REF_SPP', 1536))


def scene_dict(frame, light):
    """light: 'A', 'B' or 'AB'. frame shifts the camera ~1 px."""
    sc = mi.cornell_box()
    # camera: shift a little per frame
    dx = (frame - 1) * (2.0 / RES) * 0.9
    sc['sensor']['to_world'] = mi.ScalarTransform4f().look_at(
        origin=[dx, 0, 3.9], target=[dx, 0, 0], up=[0, 1, 0])
    sc['sensor']['film']['width'] = RES
    sc['sensor']['film']['height'] = RES
    sc['sensor']['film']['rfilter'] = {'type': 'gaussian'}
    sc['sensor']['sampler'] = {'type': 'independent', 'sample_count': SPP}
    sc['integrator'] = {'type': 'path', 'max_depth': 8}

    # smaller, brighter ceiling light -> nastier caustics
    sc['light']['to_world'] = (mi.ScalarTransform4f().translate([0, 0.99, 0.01])
                               .rotate([1, 0, 0], 90).scale([0.12, 0.12, 1]))
    sc['light']['emitter']['radiance']['value'] = [18.387 * 36, 13.9873 * 36, 6.75357 * 36]

    # checker floor (albedo texture -> tests albedo guidance)
    sc['floor']['bsdf'] = {'type': 'diffuse', 'reflectance': {
        'type': 'checkerboard', 'color0': 0.15, 'color1': 0.75,
        'to_uv': mi.ScalarTransform4f().scale([6, 6, 1])}}

    # remove the two boxes, add spheres
    sc.pop('small-box', None)
    sc.pop('large-box', None)
    sc['glass'] = {'type': 'sphere', 'center': [-0.38, -0.6, 0.2], 'radius': 0.38,
                   'bsdf': {'type': 'dielectric', 'int_ior': 1.5}}
    sc['metal'] = {'type': 'sphere', 'center': [0.45, -0.65, -0.3], 'radius': 0.33,
                   'bsdf': {'type': 'roughconductor', 'material': 'Au', 'alpha': 0.08}}
    # tiny shiny beads: legit isolated highlights (must NOT be removed)
    rng = np.random.default_rng(7)
    for i in range(14):
        x, y = rng.uniform(-0.85, 0.85), rng.uniform(0.05, 0.85)
        sc[f'bead{i}'] = {'type': 'sphere', 'center': [x, y, -0.95], 'radius': 0.012,
                          'bsdf': {'type': 'conductor', 'material': 'Ag'}}
    # thin bright emissive rod: legit 1-2 px bright line
    sc['rod'] = {'type': 'cylinder', 'p0': [-0.9, -0.2, -0.9], 'p1': [-0.2, 0.5, -0.9],
                 'radius': 0.006, 'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': [40, 40, 50]}}}
    sc['rod_bsdf_dummy'] = {'type': 'diffuse'}  # keep dict shape harmless
    sc.pop('rod_bsdf_dummy')

    # light group B: a small light on the right wall
    sc['lightB'] = {'type': 'rectangle',
                    'to_world': mi.ScalarTransform4f().translate([0.985, 0.2, 0.3])
                    .rotate([0, 1, 0], -90).scale([0.05, 0.05, 1]),
                    'bsdf': {'type': 'diffuse', 'reflectance': {'type': 'rgb', 'value': 0}},
                    'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': [300, 330, 400]}}}

    if light == 'A':   # kill B
        sc['lightB']['emitter']['radiance']['value'] = [0, 0, 0]
        sc['rod']['emitter']['radiance']['value'] = [0, 0, 0]
    elif light == 'B':  # kill A (keep the geometry so occlusion stays identical)
        sc['light']['emitter']['radiance']['value'] = [0, 0, 0]
    return sc


def render(sc, spp, seed):
    scene = mi.load_dict(sc)
    img = mi.render(scene, spp=spp, seed=seed)
    return np.array(img, dtype=np.float32)[..., :3]


def guides(frame=1):
    sc = scene_dict(frame, 'AB')
    sc['integrator'] = {'type': 'aov', 'aovs': 'alb:albedo,nn:sh_normal'}
    scene = mi.load_dict(sc)
    img = np.array(mi.render(scene, spp=64, seed=99), dtype=np.float32)
    return img[..., 0:3], img[..., 3:6]


if __name__ == '__main__':
    what = sys.argv[1:] or ['noisy', 'guides', 'ref']
    if 'noisy' in what:
        for f in range(3):
            a = render(scene_dict(f, 'A'), SPP, 1000 + f)
            b = render(scene_dict(f, 'B'), SPP, 2000 + f)
            np.save(f'{OUT}/lightA_f{f}.npy', a)
            np.save(f'{OUT}/lightB_f{f}.npy', b)
            np.save(f'{OUT}/beauty_f{f}.npy', a + b)
            print('frame', f, 'max', (a + b).max(), flush=True)
    if 'guides' in what:
        alb, nrm = guides(1)
        np.save(f'{OUT}/albedo.npy', alb)
        np.save(f'{OUT}/normal.npy', nrm)
        print('guides done', flush=True)
    if 'ref' in what:
        a = render(scene_dict(1, 'A'), REF_SPP, 5)
        b = render(scene_dict(1, 'B'), REF_SPP, 6)
        np.save(f'{OUT}/reference.npy', a + b)
        np.save(f'{OUT}/refA.npy', a)
        np.save(f'{OUT}/refB.npy', b)
        print('ref done', flush=True)
