// Test-only C API around the full pipeline.
#include "../../src/cp_pipeline.h"
#include "../../src/cp_oidn.h"
#include <cstdio>
using namespace cleanpass;
extern "C" int cp_pipeline(int w, int h, const float* cur, const float* curA, const float* curN,
                            int nn, const float** nb, const float** nbA, const float** nbN, const int* offsets,
                            const float* prm, float* outPlanar, float* matte, float* stats)
{
    PipelineParams p;
    // firefly
    p.firefly = prm[0] > 0.5f; p.ff.reference = (ReferenceMode)(int)prm[1]; p.ff.threshold = prm[2];
    p.ff.isolation = prm[3]; p.ff.method = (Method)(int)prm[4]; p.ff.temporal = prm[5] > 0.5f; p.ff.motionRadius = (int)prm[6];
    // deflicker
    p.deflicker = prm[7] > 0.5f; p.df.frames = (int)prm[8]; p.df.search = (int)prm[9]; p.df.patch = (int)prm[10];
    p.df.tolerance = prm[11]; p.df.amount = prm[12]; p.df.useGuides = prm[13] > 0.5f;
    // denoise
    p.denoise = prm[14] > 0.5f; p.denoiseAmount = prm[15];
    OidnSettings os; os.quality = (int)prm[16];
    PipelineInput in; in.width = w; in.height = h;
    in.cur.beauty = View3::interleaved(cur, w);
    if (curA) in.cur.albedo = View3::interleaved(curA, w);
    if (curN) in.cur.normal = View3::interleaved(curN, w);
    for (int k = 0; k < nn; ++k) {
        PipelineFrame f; f.beauty = View3::interleaved(nb[k], w); f.offset = offsets[k];
        if (nbA && nbA[k]) f.albedo = View3::interleaved(nbA[k], w);
        if (nbN && nbN[k]) f.normal = View3::interleaved(nbN[k], w);
        in.neighbours.push_back(f);
    }
    ReferenceFn ref = [&](int W, int H, const float* c, const View3& a, const View3& n, float* o, std::string& e) {
        return oidnPredict(os, W, H, c, a, n, o, e, std::function<bool()>()); };
    PipelineResult r; std::string err;
    if (!runPipeline(in, p, ref, ref, r, err)) { fprintf(stderr, "pipeline failed: %s\n", err.c_str()); return 1; }
    if (!r.warning.empty()) fprintf(stderr, "warning: %s\n", r.warning.c_str());
    std::copy(r.beauty.begin(), r.beauty.end(), outPlanar);
    if (matte) std::copy(r.firefly.matte.begin(), r.firefly.matte.end(), matte);
    if (stats) { for (int i = 0; i < 3; ++i) stats[i] = (float)r.seconds[i]; stats[3] = r.firefly.numCore; }
    return 0;
}
// pipeline + AOVs (planar AOV inputs, 3*w*h each; neighbour AOVs: nn per AOV, ordered like nb)
extern "C" int cp_pipeline_aov(int w, int h, const float* cur, const float* curA, const float* curN,
                            int nn, const float** nb, const int* offsets, const float* prm,
                            int na, const float** aovCur, const float** aovNb, float* outBeauty, float** aovOut)
{
    PipelineParams p;
    p.firefly = prm[0] > 0.5f; p.ff.reference = (ReferenceMode)(int)prm[1]; p.ff.threshold = prm[2];
    p.ff.isolation = prm[3]; p.ff.method = (Method)(int)prm[4]; p.ff.temporal = prm[5] > 0.5f; p.ff.motionRadius = (int)prm[6];
    p.deflicker = prm[7] > 0.5f; p.df.frames = (int)prm[8]; p.df.search = (int)prm[9]; p.df.patch = (int)prm[10];
    p.df.tolerance = prm[11]; p.df.amount = prm[12]; p.df.useGuides = prm[13] > 0.5f;
    p.denoise = prm[14] > 0.5f; p.denoiseAmount = prm[15];
    OidnSettings os; os.quality = (int)prm[16];
    PipelineInput in; in.width = w; in.height = h;
    in.cur.beauty = View3::interleaved(cur, w);
    if (curA) in.cur.albedo = View3::interleaved(curA, w);
    if (curN) in.cur.normal = View3::interleaved(curN, w);
    for (int k = 0; k < nn; ++k) { PipelineFrame f; f.beauty = View3::interleaved(nb[k], w); f.offset = offsets[k]; in.neighbours.push_back(f); }
    ReferenceFn ref = [&](int W, int H, const float* c, const View3& a, const View3& n, float* o, std::string& e) {
        return oidnPredict(os, W, H, c, a, n, o, e, std::function<bool()>()); };
    PipelineResult r; std::string err;
    if (!runPipeline(in, p, ref, ref, r, err)) return 1;
    const size_t N = (size_t)w * h;
    if (p.firefly) {
        beginAovs(r.firefly, na);
        for (int j = 0; j < na; ++j) for (int c = 0; c < 3; ++c) addAovChannel(r.firefly, p.ff, j, c, aovCur[j] + c * N, w);
        finalizeAovs(r.firefly);
    }
    std::copy(r.beauty.begin(), r.beauty.end(), outBeauty);
    for (int j = 0; j < na; ++j) {
        std::vector<std::array<const float*, 3>> nbp(nn);
        for (int k = 0; k < nn; ++k) for (int c = 0; c < 3; ++c) nbp[k][c] = aovNb[j * nn + k] + c * N;
        pipelineAov(r, p, j, {aovCur[j], aovCur[j] + N, aovCur[j] + 2 * N}, nbp, {aovOut[j], aovOut[j] + N, aovOut[j] + 2 * N});
    }
    return 0;
}
