// ASan/UBSan driver: random images incl. NaN/inf, all option combos, odd sizes.
#include "../../src/cp_core.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
using namespace cleanpass;
int main() {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    int runs = 0;
    for (int it = 0; it < 120; ++it) {
        int w = 1 + rng() % 70, h = 1 + rng() % 70; size_t N = (size_t)w * h;
        std::vector<float> b(N * 3), al(N * 3), nr(N * 3), pv(N * 3), nx(N * 3), mk(N), aov1(N), aov2(N);
        for (size_t i = 0; i < N * 3; ++i) { b[i] = U(rng) * (U(rng) < 0.03f ? 500.f : 1.f); al[i] = U(rng); nr[i] = U(rng) * 2 - 1; pv[i] = U(rng); nx[i] = U(rng); }
        for (size_t i = 0; i < N; ++i) { mk[i] = U(rng); }
        if (N > 4) { b[3] = NAN; b[7] = INFINITY; b[N * 3 - 1] = -INFINITY; }
        Params p; p.reference = ReferenceMode::Statistical;
        p.size = 1 + it % 3; p.grow = it % 4; p.method = (Method)(it % 3); p.conserveEnergy = it % 4 != 3; p.energyRadius = it % 2 ? 1 + it % 9 : 16;
        p.temporal = (it / 2) % 2; p.minLevel = (it % 5) * 0.1f; p.threads = 1 + it % 3;
        Inputs in; in.width = w; in.height = h; in.beauty = View3::interleaved(b.data(), w);
        if (it % 3) { in.albedo = View3::interleaved(al.data(), w); in.normal = View3::interleaved(nr.data(), w); }
        if (p.temporal) { in.prev = View3::interleaved(pv.data(), w); in.next = View3::interleaved(nx.data(), w); }
        if (it % 4 == 1) { in.mask = mk.data(); in.maskRowStride = w; }
        Analysis a; std::string err;
        // fake "OIDN": a blurred copy, to exercise the reference path
        ReferenceFn ref = [&](int W, int H, const float* c, const View3&, const View3&, float* o, std::string&) {
            for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) for (int k = 0; k < 3; ++k) {
                float s = 0; int n = 0;
                for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) { int xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue; s += c[((size_t)yy * W + xx) * 3 + k]; ++n; }
                o[((size_t)y * W + x) * 3 + k] = s / n; }
            return true; };
        if (it % 2) p.reference = ReferenceMode::OIDN;
        if (!analyze(in, p, ref, a, err)) { printf("fail %s\n", err.c_str()); return 1; }
        // apply on a sub-window with planar buffers and an offset
        std::vector<float> out[3]; OutBuffer ob; ob.x0 = w / 4; ob.y0 = h / 5; ob.width = w - ob.x0; ob.height = h - ob.y0; ob.rowStride = ob.width;
        for (int c = 0; c < 3; ++c) { out[c].resize((size_t)ob.width * ob.height); for (int y = 0; y < ob.height; ++y) for (int x = 0; x < ob.width; ++x) out[c][(size_t)y * ob.width + x] = b[((size_t)(y + ob.y0) * w + x + ob.x0) * 3 + c]; ob.ch[c] = out[c].data(); }
        applyLayer(a, p, -1, ob, 0.7f);
        compensateLayer(a, p, -1, ob, it % 3 ? in.albedo : View3(), 0.7f, 1 + it % 3);
        beginAovs(a, 2); addAovChannel(a, p, 0, 0, aov1.data(), w); addAovChannel(a, p, 1, 1, aov2.data(), w); finalizeAovs(a);
        applyLayer(a, p, 1, ob, 1.f);
        compensateLayer(a, p, 1, ob, in.albedo, 1.f);
        std::vector<float> m((size_t)ob.width * ob.height); OutBuffer om = ob; om.ch[0] = m.data(); om.ch[1] = om.ch[2] = nullptr; writeMatte(a, om, 1.f);
        if (p.fixNonFinite && !in.mask) for (int c = 0; c < 3; ++c) for (float v : out[c]) if (!std::isfinite(v)) { printf("nonfinite left it=%d\n", it); return 1; }
        ++runs;
    }
    printf("ok %d runs\n", runs);
}
