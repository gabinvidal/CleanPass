// Test-only C API for the deflicker stage.
#include "../../src/cp_deflicker.h"
#include <cstdio>
using namespace cleanpass;
extern "C" int cp_deflicker(int w, int h, const float* cur, const float* curA, const float* curN,
                             int nn, const float** nb, const float** nbA, const float** nbN, const int* offsets,
                             const float* prm, float* outPlanar, float* stats)
{
    DeflickerParams p; p.frames = (int)prm[0]; p.search = (int)prm[1]; p.patch = (int)prm[2];
    p.tolerance = prm[3]; p.amount = prm[4]; p.useGuides = prm[5] > 0.5f;
    DeflickerFrame c; c.beauty = View3::interleaved(cur, w);
    if (curA) c.albedo = View3::interleaved(curA, w);
    if (curN) c.normal = View3::interleaved(curN, w);
    std::vector<DeflickerFrame> nbs(nn);
    for (int k = 0; k < nn; ++k) {
        nbs[k].beauty = View3::interleaved(nb[k], w); nbs[k].offset = offsets[k];
        if (nbA && nbA[k]) nbs[k].albedo = View3::interleaved(nbA[k], w);
        if (nbN && nbN[k]) nbs[k].normal = View3::interleaved(nbN[k], w);
    }
    DeflickerResult r;
    if (!deflickerAnalyze(w, h, c, nbs, p, r)) return 1;
    const size_t N = (size_t)w * h;
    // planar channel buffers
    std::vector<float> curP(N), tmp(N);
    std::vector<std::vector<float>> nbP(nn, std::vector<float>(N));
    for (int ch = 0; ch < 3; ++ch) {
        for (size_t i = 0; i < N; ++i) curP[i] = cur[i * 3 + ch];
        std::vector<const float*> ptr; std::vector<int64_t> st;
        for (int k = 0; k < nn; ++k) { for (size_t i = 0; i < N; ++i) nbP[k][i] = nb[k][i * 3 + ch]; ptr.push_back(nbP[k].data()); st.push_back(w); }
        deflickerApplyChannel(r, p.amount, curP.data(), w, ptr, st, outPlanar + ch * N, w);
    }
    if (stats) { stats[0] = (float)r.seconds; stats[1] = r.noise; double ws = 0; for (float x : r.weight) ws += x; stats[2] = (float)(ws / std::max<size_t>(1, r.weight.size())); }
    return 0;
}
extern "C" int cp_deflicker_field(int w, int h, const float* cur, const float* curA, const float* curN,
                             int nn, const float** nb, const float** nbA, const float** nbN, const int* offsets,
                             const float* prm, float* dxo, float* dyo, float* wo)
{
    DeflickerParams p; p.frames = (int)prm[0]; p.search = (int)prm[1]; p.patch = (int)prm[2];
    p.tolerance = prm[3]; p.amount = prm[4]; p.useGuides = prm[5] > 0.5f;
    DeflickerFrame c; c.beauty = View3::interleaved(cur, w);
    if (curA) c.albedo = View3::interleaved(curA, w);
    if (curN) c.normal = View3::interleaved(curN, w);
    std::vector<DeflickerFrame> nbs(nn);
    for (int k = 0; k < nn; ++k) {
        nbs[k].beauty = View3::interleaved(nb[k], w); nbs[k].offset = offsets[k];
        if (nbA && nbA[k]) nbs[k].albedo = View3::interleaved(nbA[k], w);
        if (nbN && nbN[k]) nbs[k].normal = View3::interleaved(nbN[k], w);
    }
    DeflickerResult r;
    if (!deflickerAnalyze(w, h, c, nbs, p, r)) return 1;
    for (size_t i = 0; i < r.dx.size(); ++i) { dxo[i] = r.dx[i]; dyo[i] = r.dy[i]; wo[i] = r.weight[i]; }
    return 0;
}
