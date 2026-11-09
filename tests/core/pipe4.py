import ctypes, numpy as np, os
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get('PIPELIB', 'libpipe4.so')))
FP = ctypes.POINTER(ctypes.c_float)
class CFrame(ctypes.Structure):
    _fields_ = [(n, FP) for n in ('beauty', 'albedo', 'normal', 'depth', 'motion', 'variance')] + [('offset', ctypes.c_int)]
DEF = dict(firefly=1, reference=0, threshold=1.5, isolation=0.75, temporal=1, deflicker=0, frames=1, search=4, patch=2,
           tolerance=2.0, amount=1.0, guides=1, vectors=1, depth=1, variance=1, denoise=0, denoiseAmount=1.0, quality=0,
           stabilize=0, stabTol=0.04, denoiseAovs=0, matchBeauty=1)
ORDER = list(DEF)
INFO = ['t_ff', 't_df', 't_dn', 'numCore', 'vectorsUsed', 'vecX', 'vecY', 'vectorShare', 'identical', 'fixedSeed',
        'varianceUsed', 'noise', 'depthUsed', 'ncorr', 'stabilized']
def _c(a):
    if a is None: return None
    a = np.asarray(a, np.float32)
    if a.ndim == 2: a = np.repeat(a[..., None], 3, -1)
    if a.shape[-1] == 2: a = np.concatenate([a, np.zeros_like(a[..., :1])], -1)
    return np.ascontiguousarray(a, np.float32)
def frame(d, keep):
    arrs = {k: _c(d.get(k)) for k in ('beauty', 'albedo', 'normal', 'depth', 'motion', 'variance')}
    keep.append(arrs)
    f = CFrame(*[(arrs[k].ctypes.data_as(FP) if arrs[k] is not None else None) for k in ('beauty', 'albedo', 'normal', 'depth', 'motion', 'variance')], int(d.get('offset', 0)))
    return f
def run(cur, nbs, aovs=None, **kw):
    """aovs: list of dicts {'cur': HxWx3, 'nb': [HxWx3 per neighbour]}; returns (beauty, info, aov outs)"""
    prm = dict(DEF); prm.update(kw)
    arr = np.array([prm[k] for k in ORDER], np.float32)
    keep = []
    c = frame(cur, keep)
    n = max(1, len(nbs))
    NB = (CFrame * n)(*[frame(x, keep) for x in nbs])
    h, w = cur['beauty'].shape[:2]
    out = np.zeros((3, h, w), np.float32); info = np.zeros(len(INFO), np.float32)
    na = len(aovs) if aovs else 0
    PL = lambda a: np.ascontiguousarray(np.asarray(a, np.float32).transpose(2, 0, 1))
    ac, an, ao = [], [], []
    for a in (aovs or []):
        ac.append(PL(a['cur'])); ao.append(np.zeros((3, h, w), np.float32))
        an.extend(PL(x) for x in a['nb'])
    keep.append((ac, an, ao))
    AC = (FP * max(1, na))(*[x.ctypes.data_as(FP) for x in ac]); AN = (FP * max(1, len(an)))(*[x.ctypes.data_as(FP) for x in an])
    AO = (FP * max(1, na))(*[x.ctypes.data_as(FP) for x in ao])
    r = _lib.cp_run(w, h, ctypes.byref(c), len(nbs), NB, arr.ctypes.data_as(FP), out.ctypes.data_as(FP), info.ctypes.data_as(FP), na, AC, AN, AO)
    assert r == 0
    res = (out.transpose(1, 2, 0), dict(zip(INFO, info.tolist())))
    return res + ([x.transpose(1, 2, 0) for x in ao],) if aovs else res
def field(cur, nbs, frames=1, search=4, patch=2, tolerance=2.0, guides=1, vectors=1, depth=1, variance=1):
    keep = []; c = frame(cur, keep); n = len(nbs)
    NB = (CFrame * n)(*[frame(x, keep) for x in nbs])
    h, w = cur['beauty'].shape[:2]
    prm = np.array([frames, search, patch, tolerance, guides, vectors, depth, variance], np.float32)
    dx = np.zeros((n, h, w), np.float32); dy = np.zeros_like(dx); wt = np.zeros_like(dx); info = np.zeros(4, np.float32)
    r = _lib.cp_field4(w, h, ctypes.byref(c), n, NB, prm.ctypes.data_as(FP), dx.ctypes.data_as(FP), dy.ctypes.data_as(FP), wt.ctypes.data_as(FP), info.ctypes.data_as(FP))
    assert r == 0
    return dx, dy, wt, info
