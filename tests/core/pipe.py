import ctypes, numpy as np, os
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libpipe.so'))
FP = ctypes.POINTER(ctypes.c_float)
def _p(a): return None if a is None else a.ctypes.data_as(FP)
DEF = dict(firefly=1, reference=0, threshold=1.5, isolation=0.75, method=0, temporal=1, motionRadius=3,
           deflicker=0, frames=1, search=4, patch=2, tolerance=1.0, amount=1.0, guides=1,
           denoise=0, denoiseAmount=1.0, quality=0)
ORDER = list(DEF)
def run(cur, nbs=(), offsets=(), curA=None, curN=None, nbA=None, nbN=None, **kw):
    prm = dict(DEF); prm.update(kw)
    arr = np.array([prm[k] for k in ORDER], np.float32)
    c = lambda a: None if a is None else np.ascontiguousarray(a, np.float32)
    h, w, _ = cur.shape
    cur, curA, curN = c(cur), c(curA), c(curN)
    nbs = [c(x) for x in nbs]; nbA = [c(x) for x in nbA] if nbA else None; nbN = [c(x) for x in nbN] if nbN else None
    n = max(1, len(nbs))
    NB = (FP * n)(*[_p(x) for x in nbs]); NA = (FP * n)(*[_p(x) for x in nbA]) if nbA else None; NN = (FP * n)(*[_p(x) for x in nbN]) if nbN else None
    off = (ctypes.c_int * n)(*(list(offsets) or [0]))
    out = np.zeros((3, h, w), np.float32); matte = np.zeros((h, w), np.float32); st = np.zeros(4, np.float32)
    r = _lib.cp_pipeline(w, h, _p(cur), _p(curA), _p(curN), len(nbs), NB, NA, NN, off, _p(arr), _p(out), _p(matte), _p(st))
    assert r == 0
    return out.transpose(1, 2, 0), matte, st
def run_aov(cur, nbs, offsets, curA, curN, aovs, aovNbs, **kw):
    """aovs: list of HxWx3 at t; aovNbs[j][k]: AOV j at neighbour k"""
    prm = dict(DEF); prm.update(kw)
    arr = np.array([prm[k] for k in ORDER], np.float32)
    c = lambda a: np.ascontiguousarray(a, np.float32)
    h, w, _ = cur.shape
    cur, curA, curN = c(cur), c(curA), c(curN); nbs = [c(x) for x in nbs]
    pl = lambda a: np.ascontiguousarray(a.transpose(2, 0, 1), np.float32)
    ac = [pl(a) for a in aovs]; an = [pl(x) for j in range(len(aovs)) for x in aovNbs[j]]
    ao = [np.zeros((3, h, w), np.float32) for _ in aovs]
    n = len(nbs)
    NB = (FP * n)(*[_p(x) for x in nbs]); off = (ctypes.c_int * n)(*offsets)
    AC = (FP * len(ac))(*[_p(x) for x in ac]); AN = (FP * len(an))(*[_p(x) for x in an]); AO = (FP * len(ao))(*[_p(x) for x in ao])
    out = np.zeros((3, h, w), np.float32)
    r = _lib.cp_pipeline_aov(w, h, _p(cur), _p(curA), _p(curN), n, NB, off, _p(arr), len(ac), AC, AN, _p(out), AO)
    assert r == 0
    return out.transpose(1, 2, 0), [a.transpose(1, 2, 0) for a in ao]
