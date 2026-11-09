// C API for the Python validation of the full pipeline (motion vectors, depth,
// variance, stabilised denoise, AOV denoise).
#include "../../src/cp_pipeline.h"
#include "../../src/cp_oidn.h"
#include <cstdio>
#include <cstring>
#include <array>
#include <vector>
using namespace cleanpass;

struct CFrame
{
    const float *beauty, *albedo, *normal, *depth, *motion, *variance; // interleaved RGB (w*h*3)
    int offset;
};

static View3 V(const float* p, int w) { return p ? View3::interleaved(p, w) : View3(); }

static void fill(PipelineFrame& f, const CFrame& c, int w)
{
    f.beauty = V(c.beauty, w); f.albedo = V(c.albedo, w); f.normal = V(c.normal, w);
    f.depth = V(c.depth, w); f.motion = V(c.motion, w); f.variance = V(c.variance, w); f.offset = c.offset;
}

// prm (by index): 0 firefly 1 reference 2 threshold 3 isolation 4 temporal 5 deflicker 6 frames 7 search
// 8 patch 9 tolerance 10 amount 11 guides 12 vectors 13 depth 14 variance 15 denoise 16 denoiseAmount
// 17 quality 18 stabilize
// info: 0..2 seconds, 3 numCore, 4 vectorsUsed 5 vecX 6 vecY 7 vectorShare 8 identical 9 fixedSeed
// 10 varianceUsed 11 noise 12 depthUsed
extern "C" int cp_run(int w, int h, const CFrame* cur, int nn, const CFrame* nb, const float* prm, float* outPlanar,
                       float* info, int na, const float** aovCur, const float** aovNb, float** aovOut)
{
    PipelineParams p;
    p.firefly = prm[0] > 0.5f; p.ff.reference = (ReferenceMode)(int)prm[1]; p.ff.threshold = prm[2];
    p.ff.isolation = prm[3]; p.ff.temporal = prm[4] > 0.5f;
    p.deflicker = prm[5] > 0.5f; p.df.frames = (int)prm[6]; p.df.search = (int)prm[7]; p.df.patch = (int)prm[8];
    p.df.tolerance = prm[9]; p.df.amount = prm[10]; p.df.useGuides = prm[11] > 0.5f;
    p.df.useVectors = prm[12] > 0.5f; p.df.useDepth = prm[13] > 0.5f; p.df.useVariance = prm[14] > 0.5f;
    p.denoise = prm[15] > 0.5f; p.denoiseAmount = prm[16];
    OidnSettings os; os.quality = (int)prm[17];
    p.stabilize = prm[18] > 0.5f; p.stabilizeTolerance = prm[19]; p.denoiseAovs = prm[20] > 0.5f; p.aovsMatchBeauty = prm[21] > 0.5f;
    PipelineInput in; in.width = w; in.height = h;
    fill(in.cur, *cur, w);
    for (int k = 0; k < nn; ++k) { PipelineFrame f; fill(f, nb[k], w); in.neighbours.push_back(f); }
    ReferenceFn ref = [&](int W, int H, const float* c, const View3& a, const View3& n, float* o, std::string& e) {
        return oidnPredict(os, W, H, c, a, n, o, e, std::function<bool()>()); };
    PipelineResult r; std::string err;
    if (!runPipeline(in, p, ref, ref, r, err)) { fprintf(stderr, "pipeline failed: %s\n", err.c_str()); return 1; }
    if (!r.warning.empty()) fprintf(stderr, "warning: %s\n", r.warning.c_str());
    std::copy(r.beauty.begin(), r.beauty.end(), outPlanar);
    if (na > 0)
    {
        const size_t N = (size_t)w * h;
        std::vector<int> layers(na, -2);
        if (p.firefly) { beginAovs(r.firefly, na);
            for (int j = 0; j < na; ++j) { layers[j] = j; for (int c = 0; c < 3; ++c) addAovChannel(r.firefly, p.ff, j, c, aovCur[j] + c * N, w); }
            finalizeAovs(r.firefly); }
        std::vector<std::array<const float*, 3>> cu(na); std::vector<std::vector<std::array<const float*, 3>>> nbp(na, std::vector<std::array<const float*, 3>>(nn));
        std::vector<std::array<float*, 3>> ou(na);
        for (int j = 0; j < na; ++j) {
            cu[j] = {aovCur[j], aovCur[j] + N, aovCur[j] + 2 * N}; ou[j] = {aovOut[j], aovOut[j] + N, aovOut[j] + 2 * N};
            for (int k = 0; k < nn; ++k) nbp[j][k] = {aovNb[j * nn + k], aovNb[j * nn + k] + N, aovNb[j * nn + k] + 2 * N};
        }
        std::string wn = pipelineAovs(r, p, in, ref, layers, cu, nbp, ou);
        if (!wn.empty()) fprintf(stderr, "aov warning: %s\n", wn.c_str());
    }
    if (info)
    {
        for (int i = 0; i < 3; ++i) info[i] = (float)r.seconds[i];
        info[3] = (float)r.firefly.numCore;
        const DeflickerResult& d = r.deflicker;
        info[4] = d.vectorsUsed; info[5] = d.vecX; info[6] = d.vecY; info[7] = d.vectorShare;
        info[8] = d.identical; info[9] = d.fixedSeed; info[10] = d.varianceUsed; info[11] = d.noise; info[12] = d.depthUsed; info[13] = d.noiseCorrelation; info[14] = r.stabilized;
    }
    return 0;
}
extern "C" void cp_refine(int w, int h, const float* a, const float* aA, const float* aN, const float* b, const float* bA,
                           const float* bN, float* dx, float* dy)
{
    DeflickerFrame A, B;
    A.beauty = V(a, w); A.albedo = V(aA, w); A.normal = V(aN, w);
    B.beauty = V(b, w); B.albedo = V(bA, w); B.normal = V(bN, w);
    deflickerRefine(w, h, A, B, dx, dy);
}
// deflicker field with all the optional passes: cur + nn neighbours (CFrame), outputs count*w*h
extern "C" int cp_field4(int w, int h, const CFrame* cur, int nn, const CFrame* nb, const float* prm, float* dx, float* dy,
                          float* wt, float* info)
{
    DeflickerParams p; p.frames = (int)prm[0]; p.search = (int)prm[1]; p.patch = (int)prm[2]; p.tolerance = prm[3];
    p.useGuides = prm[4] > 0.5f; p.useVectors = prm[5] > 0.5f; p.useDepth = prm[6] > 0.5f; p.useVariance = prm[7] > 0.5f;
    DeflickerFrame c; c.beauty = V(cur->beauty, w); c.albedo = V(cur->albedo, w); c.normal = V(cur->normal, w);
    c.depth = V(cur->depth, w); c.motion = V(cur->motion, w); c.variance = V(cur->variance, w);
    std::vector<DeflickerFrame> nbs;
    for (int k = 0; k < nn; ++k) { DeflickerFrame f; f.beauty = V(nb[k].beauty, w); f.albedo = V(nb[k].albedo, w);
        f.normal = V(nb[k].normal, w); f.depth = V(nb[k].depth, w); f.offset = nb[k].offset; nbs.push_back(f); }
    DeflickerResult r;
    if (!deflickerAnalyze(w, h, c, nbs, p, r)) return 1;
    std::copy(r.dx.begin(), r.dx.end(), dx); std::copy(r.dy.begin(), r.dy.end(), dy); std::copy(r.weight.begin(), r.weight.end(), wt);
    info[0] = r.vectorsUsed; info[1] = r.vecX; info[2] = r.vecY; info[3] = r.vectorShare;
    return 0;
}
