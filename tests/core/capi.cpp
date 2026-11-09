// Test-only C API around the core, for driving it from Python (ctypes).
#include "../../src/cp_core.h"
#include "../../src/cp_oidn.h"
#include <cstring>
#include <cstdio>

using namespace cleanpass;

extern "C" int cp_run(int w, int h, const float* beauty, const float* albedo, const float* normal,
                       const float* prev, const float* next, const float* mask, const float* prm,
                       int nAovs, const float** aovsPlanar, // each 3*w*h planar
                       float* outBeauty /*3*w*h planar*/, float** outAovs, float* outMatte, float* stats)
{
    Params p;
    p.reference = (ReferenceMode)(int)prm[0];
    p.threshold = prm[1]; p.softness = prm[2]; p.isolation = prm[3]; p.adaptivity = prm[4];
    p.minLevel = prm[5]; p.size = (int)prm[6]; p.grow = (int)prm[7]; p.fixNonFinite = prm[8] > 0.5f;
    p.method = (Method)(int)prm[9]; p.conserveEnergy = prm[10] > 0.5f; p.energyRadius = prm[11];
    p.energyAmount = prm[12]; p.temporal = prm[13] > 0.5f; p.motionRadius = (int)prm[14];
    float mix = prm[15];
    OidnSettings os; os.quality = (int)prm[16]; p.preCleanBias = prm[17];

    Inputs in;
    in.width = w; in.height = h;
    in.beauty = View3::interleaved(beauty, w);
    if (albedo) in.albedo = View3::interleaved(albedo, w);
    if (normal) in.normal = View3::interleaved(normal, w);
    if (prev) in.prev = View3::interleaved(prev, w);
    if (next) in.next = View3::interleaved(next, w);
    if (mask) { in.mask = mask; in.maskRowStride = w; }

    ReferenceFn ref = [&](int W, int H, const float* color, const View3& a, const View3& n, float* out, std::string& err) {
        return oidnPredict(os, W, H, color, a, n, out, err, std::function<bool()>());
    };
    Analysis an;
    std::string err;
    if (!analyze(in, p, ref, an, err)) { fprintf(stderr, "analyze failed: %s\n", err.c_str()); return 1; }
    if (!an.warning.empty()) fprintf(stderr, "warning: %s\n", an.warning.c_str());

    const size_t N = (size_t)w * h;
    for (int c = 0; c < 3; ++c)
        for (size_t i = 0; i < N; ++i) outBeauty[c * N + i] = beauty[i * 3 + c];
    OutBuffer ob; ob.width = w; ob.height = h; ob.rowStride = w;
    for (int c = 0; c < 3; ++c) ob.ch[c] = outBeauty + c * N;
    applyLayer(an, p, -1, ob, mix);

    if (nAovs > 0) {
        beginAovs(an, nAovs);
        for (int j = 0; j < nAovs; ++j)
            for (int c = 0; c < 3; ++c) addAovChannel(an, p, j, c, aovsPlanar[j] + c * N, w);
        finalizeAovs(an);
        for (int j = 0; j < nAovs; ++j) {
            memcpy(outAovs[j], aovsPlanar[j], 3 * N * sizeof(float));
            OutBuffer oa = ob;
            for (int c = 0; c < 3; ++c) oa.ch[c] = outAovs[j] + c * N;
            applyLayer(an, p, j, oa, mix);
        }
    }
    if (outMatte) {
        memset(outMatte, 0, N * sizeof(float));
        OutBuffer om = ob; om.ch[0] = outMatte; om.ch[1] = om.ch[2] = nullptr;
        writeMatte(an, om, 1.f);
    }
    if (stats) { stats[0] = an.numCore; stats[1] = (float)an.hitIndex.size(); stats[2] = (float)an.seconds; }
    return 0;
}
