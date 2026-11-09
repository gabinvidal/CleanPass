// Reuse check: frame t computed normally must equal frame t computed from the
// result the previous frame's stabilisation produced for it (precomputed).
#include "../../src/cp_pipeline.h"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
using namespace cleanpass;
const int W = 96, H = 64;
struct Fr { std::vector<float> b, a, n, z, m; };
Fr make(int f)
{
    Fr r; std::mt19937 rng(100 + f); std::uniform_real_distribution<float> U(0, 1);
    r.b.resize(W * H * 3); r.a.resize(W * H * 3); r.n.resize(W * H * 3); r.z.resize(W * H * 3); r.m.resize(W * H * 3, 0.f);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        size_t i = (size_t)(y * W + x) * 3; float xs = x - 1.5f * f;
        float base = 0.3f + 0.2f * std::sin(xs * 0.2f) * std::cos(y * 0.15f);
        for (int c = 0; c < 3; ++c) { r.b[i + c] = base * (0.5f + U(rng)) * (U(rng) < 0.01f ? 30.f : 1.f); r.a[i + c] = 0.5f + 0.3f * std::sin(xs * 0.2f); r.n[i + c] = c == 2 ? 1.f : 0.f; r.z[i + c] = 5.f; }
        r.m[i] = 1.5f; r.m[i + 1] = 0.f;
    }
    return r;
}
static View3 V(std::vector<float>& v) { return View3::interleaved(v.data(), W); }
PipelineFrame pf(Fr& f, int off, bool mv) { PipelineFrame p; p.beauty = V(f.b); p.albedo = V(f.a); p.normal = V(f.n); p.depth = V(f.z); if (mv) p.motion = V(f.m); p.offset = off; return p; }
int main()
{
    std::vector<Fr> F; for (int f = 0; f < 8; ++f) F.push_back(make(f));
    ReferenceFn blur = [](int w, int h, const float* c, const View3&, const View3&, float* o, std::string&) {
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) for (int k = 0; k < 3; ++k) { float s = 0; int n = 0;
            for (int dy = -2; dy <= 2; ++dy) for (int dx = -2; dx <= 2; ++dx) { int xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue; s += c[((size_t)yy * w + xx) * 3 + k]; ++n; }
            o[((size_t)y * w + x) * 3 + k] = s / n; } return true; };
    for (int mode = 0; mode < 2; ++mode) {
        PipelineParams p; p.firefly = true; p.ff.reference = ReferenceMode::Statistical; p.deflicker = mode == 0; p.df.frames = 1; p.denoise = true; p.stabilize = true;
        auto input = [&](int t) { PipelineInput in; in.width = W; in.height = H; in.cur = pf(F[t], 0, true);
            for (int o : {-1, 1, -2, 2}) in.neighbours.push_back(pf(F[t + o], o, std::abs(o) == 1)); return in; };
        std::string err;
        PipelineResult a; PipelineInput ia = input(4); if (!runPipeline(ia, p, blur, blur, a, err)) { printf("fail %s\n", err.c_str()); return 1; }
        PipelineResult prev; PipelineInput ip = input(3); if (!runPipeline(ip, p, blur, blur, prev, err) || !prev.subNext) { printf("no subNext\n"); return 1; }
        PipelineInput ib = input(4); ib.precomputed = prev.subNext.get();
        // D(t-1): what frame 3 denoised for itself (its stage-3 result)
        ib.denoisedPrev = &prev.beautyDenoised;
        PipelineResult b; if (!runPipeline(ib, p, blur, blur, b, err)) { printf("fail b\n"); return 1; }
        double md = 0; for (size_t i = 0; i < a.beauty.size(); ++i) md = std::max(md, (double)std::fabs(a.beauty[i] - b.beauty[i]));
        double mf = 0; for (size_t i = 0; i < a.beautyDenoised.size(); ++i) mf = std::max(mf, (double)std::fabs(a.beautyDenoised[i] - b.beautyDenoised[i]));
        if (a.subPrev) { double q=0; for (size_t i=0;i<a.subPrev->beauty.size();++i) q=std::max(q,(double)std::fabs(a.subPrev->beauty[i]-prev.beautyDenoised[i])); printf("  subPrev(a) vs prev.beautyDenoised %.3g\n", q);} 
        { double q=0; for (size_t i=0;i<a.deflicker.dx.size();++i) q=std::max(q,(double)std::fabs(a.deflicker.dx[i]-b.deflicker.dx[i])); printf("  dx diff %.3g count %d %d\n", q, a.deflicker.count, b.deflicker.count);} 
        printf("mode %s: max |normal - reused| final %.3g, denoised %.3g, stabilized %d/%d, subNext computed in b: %d\n", mode == 0 ? "deflicker+denoise+stab" : "denoise+stab", md, mf, a.stabilized, b.stabilized, (int)(bool)b.subNext);
    }
    // chain: frames 2..5 in order, each using what the previous one prepared
    {
        PipelineParams p; p.firefly = true; p.ff.reference = ReferenceMode::Statistical; p.deflicker = true; p.df.frames = 1; p.denoise = true; p.stabilize = true;
        auto input = [&](int t) { PipelineInput in; in.width = W; in.height = H; in.cur = pf(F[t], 0, true);
            for (int o : {-1, 1, -2, 2}) { in.neighbours.push_back(pf(F[t + o], o, std::abs(o) == 1)); } return in; };
        std::string err; std::shared_ptr<PipelineResult> nextFull; std::vector<float> curDen; double worst = 0; int reusedN = 0;
        for (int t = 2; t <= 5; ++t) {
            PipelineResult ref; PipelineInput ir = input(t); runPipeline(ir, p, blur, blur, ref, err);
            PipelineInput ic = input(t); if (nextFull) ic.precomputed = nextFull.get(); if (!curDen.empty()) ic.denoisedPrev = &curDen;
            PipelineResult c; runPipeline(ic, p, blur, blur, c, err);
            double md = 0; for (size_t i = 0; i < ref.beauty.size(); ++i) md = std::max(md, (double)std::fabs(ref.beauty[i] - c.beauty[i]));
            worst = std::max(worst, md); reusedN += c.reused; nextFull = c.subNext; curDen = c.beautyDenoised;
        }
        printf("chain 2..5: worst |normal - chained| %.3g, reused %d/4\n", worst, reusedN);
    }
}
