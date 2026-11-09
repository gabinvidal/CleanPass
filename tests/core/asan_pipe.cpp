#include "../../src/cp_pipeline.h"
#include <cmath>
#include <cstdio>
#include <random>
using namespace cleanpass;
int main() {
    std::mt19937 rng(3); std::uniform_real_distribution<float> U(0.f, 1.f);
    int runs = 0;
    for (int it = 0; it < 160; ++it) {
        int w = 1 + rng() % 90, h = 1 + rng() % 90; if (it % 7 == 0) { w = 64 + rng() % 40; h = 64 + rng() % 40; }
        size_t N = (size_t)w * h;
        auto img = [&]() { std::vector<float> v(N * 3); for (auto& x : v) x = U(rng) * (U(rng) < 0.02f ? 300.f : 1.f); return v; };
        std::vector<std::vector<float>> fr(5); for (auto& f : fr) f = img();
        std::vector<float> al(N * 3), nr(N * 3); for (auto& x : al) x = U(rng); for (auto& x : nr) x = U(rng) * 2 - 1;
        // motion vectors (random convention / scale, some NaN), depth (some inf / 0 / negative), variance
        std::vector<float> mv(N * 3), z(N * 3), var(N * 3);
        const float sc = (it % 3 == 0) ? 1.f / w : (it % 3 == 1 ? 1.f : -0.5f);
        for (size_t i = 0; i < N; ++i) { mv[i * 3] = (U(rng) * 6 - 3) * sc; mv[i * 3 + 1] = (U(rng) * 6 - 3) * sc; mv[i * 3 + 2] = 0;
            float zz = 1 + U(rng) * 10; if (U(rng) < 0.03f) zz = INFINITY; if (U(rng) < 0.02f) zz = 0; if (U(rng) < 0.02f) zz = -zz;
            z[i * 3] = z[i * 3 + 1] = z[i * 3 + 2] = zz; for (int c = 0; c < 3; ++c) var[i * 3 + c] = U(rng) * 0.1f; }
        if (N > 10) { mv[3] = NAN; var[7] = INFINITY; }
        if (it % 9 == 4) for (size_t i = 0; i < N * 3; ++i) fr[1][i] = fr[0][i]; // fixed seed / still
        if (N > 6) { fr[0][2] = NAN; fr[1][5] = INFINITY; }
        PipelineParams p; p.firefly = it % 3 != 0; p.ff.reference = ReferenceMode::Statistical; p.ff.temporal = it % 2;
        p.deflicker = it % 4 != 1; p.df.frames = 1 + it % 2; p.df.search = 1 + it % 5; p.df.patch = 1 + it % 4; p.df.useGuides = it % 2;
        p.denoise = it % 5 == 0 || it % 7 == 3; p.ff.conserveEnergy = it % 3 == 1;
        p.stabilize = it % 2 == 0; p.denoiseAovs = it % 3 == 0; p.aovsMatchBeauty = it % 4 != 3;
        PipelineInput in; in.width = w; in.height = h; in.cur.beauty = View3::interleaved(fr[0].data(), w);
        if (it % 2) { in.cur.albedo = View3::interleaved(al.data(), w); in.cur.normal = View3::interleaved(nr.data(), w); }
        if (it % 4 < 2) in.cur.motion = View3::interleaved(mv.data(), w);
        if (it % 5 < 3) in.cur.depth = View3::interleaved(z.data(), w);
        if (it % 3 < 2) in.cur.variance = View3::interleaved(var.data(), w);
        const int offs[4] = {-1, 1, -2, 2};
        for (int k = 0; k < 4; ++k) { if (it % 6 == 5 && k == 1) continue; PipelineFrame f; f.beauty = View3::interleaved(fr[k + 1].data(), w); f.offset = offs[k];
            if (it % 2) { f.albedo = in.cur.albedo; f.normal = in.cur.normal; } f.depth = in.cur.depth; f.motion = in.cur.motion; in.neighbours.push_back(f); }
        ReferenceFn den = [&](int W, int H, const float* c, const View3&, const View3&, float* o, std::string&) { for (size_t i = 0; i < (size_t)W * H * 3; ++i) o[i] = c[i] * 0.9f; return true; };
        PipelineResult r; std::string err;
        if (!runPipeline(in, p, den, den, r, err)) { printf("fail %s\n", err.c_str()); return 1; }
        // AOVs
        std::vector<float> a1(N * 3), a2(N * 3); for (auto& x : a1) x = U(rng); for (auto& x : a2) x = U(rng);
        if (p.firefly) { beginAovs(r.firefly, 1); for (int c = 0; c < 3; ++c) addAovChannel(r.firefly, p.ff, 0, c, a1.data() + c * N, w); finalizeAovs(r.firefly); }
        std::vector<std::array<const float*, 3>> nb(in.neighbours.size());
        for (auto& q : nb) q = {a2.data(), a2.data() + N, a2.data() + 2 * N};
        std::vector<float> o(N * 3);
        pipelineAov(r, p, 0, {a1.data(), a1.data() + N, a1.data() + 2 * N}, nb, {o.data(), o.data() + N, o.data() + 2 * N});
        {   // all AOVs at once (with a complete pair: a1 + (beauty - a1))
            std::vector<float> a3(N * 3); for (size_t i = 0; i < N * 3; ++i) { const int c = (int)(i % 3); a3[i] = 0.f; (void)c; }
            for (int c = 0; c < 3; ++c) for (size_t i = 0; i < N; ++i) a3[c * N + i] = (it % 2 ? fr[0][i * 3 + c] : 0.f) - a1[c * N + i];
            std::vector<float> o1(N * 3), o2(N * 3);
            std::vector<std::vector<std::array<const float*, 3>>> nbs2 = {nb, nb};
            std::string wn = pipelineAovs(r, p, in, den, {p.firefly ? 0 : -2, -2}, {{a1.data(), a1.data() + N, a1.data() + 2 * N}, {a3.data(), a3.data() + N, a3.data() + 2 * N}}, nbs2,
                                          {{o1.data(), o1.data() + N, o1.data() + 2 * N}, {o2.data(), o2.data() + N, nullptr}});
        }
        for (float v : r.beauty) if (!std::isfinite(v) && p.firefly && p.ff.fixNonFinite) { printf("nonfinite it=%d\n", it); return 1; }
        ++runs;
    }
    printf("ok %d runs\n", runs);
}
