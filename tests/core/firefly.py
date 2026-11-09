import ctypes, numpy as np, os, time
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libfirefly.so'))
FP = ctypes.POINTER(ctypes.c_float)
def _p(a): return None if a is None else a.ctypes.data_as(FP)
DEF = dict(reference=0, threshold=1.5, softness=0.75, isolation=0.75, adaptivity=0.25, minLevel=0.0,
           size=2, grow=1, fixNonFinite=1, method=0, conserve=0, energyRadius=8, energyAmount=1,
           temporal=0, motionRadius=2, mix=1.0, quality=0, preClean=1.0)
ORDER = list(DEF.keys())
def run(beauty, albedo=None, normal=None, prev=None, next=None, mask=None, aovs=(), **kw):
    prm = dict(DEF); prm.update(kw)
    arr = np.array([prm[k] for k in ORDER], np.float32)
    h, w, _ = beauty.shape
    c = lambda a: None if a is None else np.ascontiguousarray(a, np.float32)
    beauty, albedo, normal, prev, next, mask = map(c, (beauty, albedo, normal, prev, next, mask))
    out = np.zeros((3, h, w), np.float32); matte = np.zeros((h, w), np.float32); stats = np.zeros(3, np.float32)
    ap = [np.ascontiguousarray(a.transpose(2, 0, 1), np.float32) for a in aovs]
    ao = [np.zeros((3, h, w), np.float32) for _ in aovs]
    AP = (FP * max(1, len(ap)))(*[_p(a) for a in ap]); AO = (FP * max(1, len(ao)))(*[_p(a) for a in ao])
    r = _lib.cp_run(w, h, _p(beauty), _p(albedo), _p(normal), _p(prev), _p(next), _p(mask), _p(arr),
                     len(ap), AP, _p(out), AO, _p(matte), _p(stats))
    assert r == 0
    return out.transpose(1, 2, 0), [a.transpose(1, 2, 0) for a in ao], matte, stats
