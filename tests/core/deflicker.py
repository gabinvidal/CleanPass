import ctypes, numpy as np, os
_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libdeflicker.so'))
FP = ctypes.POINTER(ctypes.c_float)
def _p(a): return None if a is None else a.ctypes.data_as(FP)
def run(cur, nbs, offsets, curA=None, curN=None, nbA=None, nbN=None, frames=1, search=4, patch=2, tolerance=1.0, amount=1.0, guides=1):
    c = lambda a: None if a is None else np.ascontiguousarray(a, np.float32)
    h, w, _ = cur.shape
    cur, curA, curN = c(cur), c(curA), c(curN)
    nbs = [c(x) for x in nbs]; nbA = [c(x) for x in nbA] if nbA else None; nbN = [c(x) for x in nbN] if nbN else None
    NB = (FP * len(nbs))(*[_p(x) for x in nbs])
    NA = (FP * len(nbs))(*[_p(x) for x in nbA]) if nbA else None
    NN = (FP * len(nbs))(*[_p(x) for x in nbN]) if nbN else None
    off = (ctypes.c_int * len(nbs))(*offsets)
    prm = np.array([frames, search, patch, tolerance, amount, guides], np.float32)
    out = np.zeros((3, h, w), np.float32); st = np.zeros(3, np.float32)
    r = _lib.cp_deflicker(w, h, _p(cur), _p(curA), _p(curN), len(nbs), NB, NA, NN, off, _p(prm), _p(out), _p(st))
    assert r == 0
    return out.transpose(1, 2, 0), st
def field(cur, nbs, offsets, curA=None, curN=None, nbA=None, nbN=None, frames=1, search=4, patch=2, tolerance=1.0, amount=1.0, guides=1):
    c = lambda a: None if a is None else np.ascontiguousarray(a, np.float32)
    h, w, _ = cur.shape
    cur, curA, curN = c(cur), c(curA), c(curN)
    nbs = [c(x) for x in nbs]; nbA = [c(x) for x in nbA] if nbA else None; nbN = [c(x) for x in nbN] if nbN else None
    NB = (FP * len(nbs))(*[_p(x) for x in nbs])
    NA = (FP * len(nbs))(*[_p(x) for x in nbA]) if nbA else None
    NN = (FP * len(nbs))(*[_p(x) for x in nbN]) if nbN else None
    off = (ctypes.c_int * len(nbs))(*offsets)
    prm = np.array([frames, search, patch, tolerance, amount, guides], np.float32)
    n = len(nbs) * w * h
    dx = np.zeros(n, np.float32); dy = np.zeros(n, np.float32); wt = np.zeros(n, np.float32)
    r = _lib.cp_deflicker_field(w, h, _p(cur), _p(curA), _p(curN), len(nbs), NB, NA, NN, off, _p(prm),
        _p(dx), _p(dy), _p(wt))
    return dx.reshape(len(nbs), h, w), dy.reshape(len(nbs), h, w), wt.reshape(len(nbs), h, w)
