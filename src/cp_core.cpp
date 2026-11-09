// CleanPass - core algorithm
// Copyright (c) 2026 Gabin Vidal. MIT License (see LICENSE).

#include "cp_core.h"
#include "cp_lowpass.h"
#include "cp_parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <thread>
#include <utility>

namespace cleanpass
{
namespace
{

const float kEps = 1e-4f;      // absolute epsilon for energy ratios
const float kMinHit = 1e-4f;   // matte values below this are ignored
const int kRankHigh = 3;       // "ring high" = 3rd largest neighbour

// ---------------------------------------------------------------------------
// small utilities
// ---------------------------------------------------------------------------

inline float smoothstep(float a, float b, float x)
{
    if (b <= a)
        return x >= a ? 1.f : 0.f;
    float t = (x - a) / (b - a);
    t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
    return t * t * (3.f - 2.f * t);
}

inline float stops(float a, float b) // log2 ratio, robust to zero
{
    return std::log2((a + kEps) / (b + kEps));
}

inline bool finite3(float r, float g, float b)
{
    return std::isfinite(r) && std::isfinite(g) && std::isfinite(b);
}

inline float finiteOr(float v, float fallback) { return std::isfinite(v) ? v : fallback; }

int threadCount(const Params& p) { return detail::threadCount(p.threads); }
using detail::parallelFor;

// Ring of neighbours: Chebyshev distance in [size, size + 1].
std::vector<std::pair<int, int>> ringOffsets(int size)
{
    std::vector<std::pair<int, int>> o;
    const int r = size + 1;
    for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx)
        {
            const int d = std::max(std::abs(dx), std::abs(dy));
            if (d >= size && d <= r)
                o.emplace_back(dx, dy);
        }
    return o;
}

// ---------------------------------------------------------------------------
// guide signature: 3 x int8 normal + 8-bit log albedo luminance
// ---------------------------------------------------------------------------

inline uint32_t packSig(const float n[3], bool hasN, float alb, bool hasA)
{
    uint32_t s = 0;
    if (hasN)
    {
        float l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        float k = l > 1e-6f ? 127.f / l : 0.f;
        for (int c = 0; c < 3; ++c)
        {
            int q = (int)std::lround(finiteOr(n[c] * k, 0.f));
            q = std::max(-127, std::min(127, q));
            s |= (uint32_t)(uint8_t)(int8_t)q << (8 * c);
        }
    }
    if (hasA)
    {
        float a = std::max(0.f, finiteOr(alb, 0.f));
        int q = (int)std::lround((std::log2(a + 1e-3f) + 10.f) * 12.75f);
        q = std::max(0, std::min(255, q));
        s |= (uint32_t)q << 24;
    }
    return s;
}

inline float sigSimilarity(uint32_t a, uint32_t b, bool hasN, bool hasA)
{
    float w = 1.f;
    if (hasN)
    {
        int dot = 0, la = 0, lb = 0;
        for (int c = 0; c < 3; ++c)
        {
            int na = (int)(int8_t)((a >> (8 * c)) & 0xff);
            int nb = (int)(int8_t)((b >> (8 * c)) & 0xff);
            dot += na * nb; la += na * na; lb += nb * nb;
        }
        if (la > 0 && lb > 0)
        {
            float cosv = (float)dot / std::sqrt((float)la * (float)lb);
            w *= std::exp(-(1.f - cosv) / 0.1f);
        }
    }
    if (hasA)
    {
        float d = (float)((int)(a >> 24) - (int)(b >> 24)) / 12.75f; // stops
        w *= std::exp(-(d * d) / 0.5f);
    }
    return std::max(w, 1e-3f);
}

// ---------------------------------------------------------------------------
// working context
// ---------------------------------------------------------------------------

struct Ctx
{
    const Inputs& in;
    const Params& p;
    int W, H;
    std::vector<float> E;        // energy, non-finite -> 0
    std::vector<uint8_t> bad;    // non-finite pixel
    std::vector<uint32_t> sig;   // guide signature
    bool hasN = false, hasA = false;
    std::vector<std::pair<int, int>> rings[4]; // index = scale (1..3)

    Ctx(const Inputs& i, const Params& pp) : in(i), p(pp), W(i.width), H(i.height) {}

    float px(int idx, int c) const { return in.beauty.at(idx % W, idx / W, c); }

    float simil(int a, int b) const
    {
        if (sig.empty())
            return 1.f;
        return sigSimilarity(sig[a], sig[b], hasN, hasA);
    }

    // Guided weighted median of the ring energies ("medoid"): returns the index
    // of an actual neighbour sample of median brightness among the neighbours
    // that look like the same surface. `excl` marks pixels to skip.
    int medoid(int idx, const std::vector<uint8_t>& excl, int scale, const float* En = nullptr) const
    {
        if (!En)
            En = E.data();
        const auto& ring = rings[scale];
        const int x = idx % W, y = idx / W;
        struct S { float e, w; int q; };
        S buf[64];
        int n = 0;
        float wsum = 0.f;
        for (int pass = 0; pass < 2 && n < 3; ++pass)
        {
            n = 0; wsum = 0.f;
            for (const auto& o : ring)
            {
                const int xx = x + o.first, yy = y + o.second;
                if (xx < 0 || yy < 0 || xx >= W || yy >= H)
                    continue;
                const int q = yy * W + xx;
                if (bad[q] || (pass == 0 && excl[q]))
                    continue;
                const float w = simil(idx, q);
                buf[n++] = {En[q], w, q};
                wsum += w;
            }
        }
        if (n == 0)
            return -1;
        std::sort(buf, buf + n, [](const S& a, const S& b) { return a.e < b.e; });
        float acc = 0.f;
        const float half = 0.5f * wsum;
        for (int i = 0; i < n; ++i)
        {
            acc += buf[i].w;
            if (acc >= half)
                return buf[i].q;
        }
        return buf[n - 1].q;
    }

    // Per-channel median of the ring, skipping `excl` pixels and non-finite values.
    bool ringMedian(int idx, const std::vector<uint8_t>& excl, int scale, float med[3]) const
    {
        const auto& ring = rings[scale];
        const int x = idx % W, y = idx / W;
        float v[3][64];
        int n[3] = {0, 0, 0};
        for (const auto& o : ring)
        {
            const int xx = x + o.first, yy = y + o.second;
            if (xx < 0 || yy < 0 || xx >= W || yy >= H)
                continue;
            const int q = yy * W + xx;
            if (excl[q])
                continue;
            for (int c = 0; c < 3; ++c)
            {
                const float f = in.beauty.at(xx, yy, c);
                if (std::isfinite(f))
                    v[c][n[c]++] = f;
            }
        }
        bool ok = true;
        for (int c = 0; c < 3; ++c)
        {
            if (n[c] == 0)
            {
                ok = false;
                med[c] = 0.f;
                continue;
            }
            std::nth_element(v[c], v[c] + n[c] / 2, v[c] + n[c]);
            med[c] = v[c][n[c] / 2];
        }
        return ok;
    }
};

struct Cand
{
    int idx;
    int scale;    // ring scale that isolates it best
    float diso;   // stops above the ring high
    float hi;     // ring high energy
    float S;      // local noise spread (stops)
    float eref;   // expected energy (statistical)
    int med;      // medoid sample index
    float m;      // matte
};

// k-th largest energy of a (2r+1)^2 window in a beauty view (temporal)
float windowHigh(const View3& v, int W, int H, int x, int y, int r, float* top)
{
    for (int k = 0; k < kRankHigh; ++k)
        top[k] = 0.f;
    for (int dy = -r; dy <= r; ++dy)
    {
        const int yy = y + dy;
        if (yy < 0 || yy >= H)
            continue;
        for (int dx = -r; dx <= r; ++dx)
        {
            const int xx = x + dx;
            if (xx < 0 || xx >= W)
                continue;
            float r_ = v.at(xx, yy, 0), g_ = v.at(xx, yy, 1), b_ = v.at(xx, yy, 2);
            if (!finite3(r_, g_, b_))
                continue;
            float e = energyOf(r_, g_, b_);
            if (e > top[kRankHigh - 1])
            {
                int k = kRankHigh - 1;
                while (k > 0 && top[k - 1] < e)
                {
                    top[k] = top[k - 1];
                    --k;
                }
                top[k] = e;
            }
        }
    }
    return top[kRankHigh - 1];
}

} // namespace

// ===========================================================================
// analyze
// ===========================================================================

bool analyze(const Inputs& in, const Params& pIn, const ReferenceFn& reference,
             Analysis& out, std::string& err, const AbortFn& aborted)
{
    const auto t0 = std::chrono::steady_clock::now();
    out = Analysis();
    if (in.width <= 0 || in.height <= 0 || !in.beauty.valid())
    {
        err = "no input image";
        return false;
    }

    Params p = pIn;
    p.size = std::max(1, std::min(3, p.size));
    p.grow = std::max(0, std::min(3, p.grow));
    p.softness = std::max(0.f, p.softness);
    p.motionRadius = std::max(0, std::min(16, p.motionRadius));

    Ctx cx(in, p);
    const int W = cx.W, H = cx.H, N = W * H;
    const int nth = threadCount(p);
    auto isAborted = [&]() { return aborted && aborted(); };

    out.width = W;
    out.height = H;

    // -- energy, non-finite map, guide signature ---------------------------
    cx.E.assign(N, 0.f);
    cx.bad.assign(N, 0);
    cx.hasN = in.normal.valid();
    cx.hasA = in.albedo.valid();
    if (cx.hasN || cx.hasA)
        cx.sig.assign(N, 0u);
    parallelFor(H, nth, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                const int i = y * W + x;
                const float r = in.beauty.at(x, y, 0), g = in.beauty.at(x, y, 1), b = in.beauty.at(x, y, 2);
                if (!finite3(r, g, b))
                {
                    cx.bad[i] = 1;
                    cx.E[i] = energyOf(finiteOr(r, 0.f), finiteOr(g, 0.f), finiteOr(b, 0.f));
                }
                else
                    cx.E[i] = energyOf(r, g, b);
                if (!cx.sig.empty())
                {
                    float n[3] = {0, 0, 0}, a = 0.f;
                    if (cx.hasN)
                        for (int c = 0; c < 3; ++c)
                            n[c] = in.normal.at(x, y, c);
                    if (cx.hasA)
                        a = 0.2126f * in.albedo.at(x, y, 0) + 0.7152f * in.albedo.at(x, y, 1) +
                            0.0722f * in.albedo.at(x, y, 2);
                    cx.sig[i] = packSig(n, cx.hasN, a, cx.hasA);
                }
            }
    });
    if (isAborted())
        return false;

    for (int s = 1; s <= 3; ++s)
        cx.rings[s] = ringOffsets(s);
    const float levelLo = p.minLevel * 0.5f, levelHi = p.minLevel;

    // Matte of a candidate. A pixel that is extremely far above its expected
    // value (threshold + 3 stops) no longer needs to be perfectly isolated:
    // that is what lets dense clusters of fireflies go away.
    auto matteOf = [&](float e, float eref, float S, float diso) {
        const float teff = p.threshold + p.adaptivity * S;
        const float dref = stops(e, eref);
        const float iso = std::max(smoothstep(p.isolation, p.isolation + p.softness, diso),
                                   smoothstep(teff + 3.f, teff + 3.f + p.softness, dref));
        return smoothstep(teff, teff + p.softness, dref) * iso * smoothstep(levelLo, levelHi, e);
    };

    // -- candidate discovery ---------------------------------------------------
    // Isolation test (multi-scale) on a set of pixels, using energy array En.
    auto isolationPass = [&](const std::vector<int>* subset, const float* En, const std::vector<uint8_t>& skip) {
        const int count = subset ? (int)subset->size() : N;
        std::vector<std::vector<Cand>> perThread(nth);
        const int chunk = (count + nth - 1) / nth;
        std::vector<std::thread> pool;
        for (int t = 0; t < nth; ++t)
        {
            const int b0 = t * chunk, b1 = std::min(count, b0 + chunk);
            if (b0 >= b1)
                break;
            pool.emplace_back([&, t, b0, b1]() {
                auto& list = perThread[t];
                float top[kRankHigh];
                for (int k = b0; k < b1; ++k)
                {
                    const int i = subset ? (*subset)[k] : k;
                    if (skip[i])
                        continue;
                    const int x = i % W, y = i / W;
                    if (cx.bad[i])
                    {
                        if (p.fixNonFinite)
                            list.push_back({i, p.size, 99.f, 0.f, 0.f, 0.f, -1, 1.f});
                        continue;
                    }
                    const float e = En[i];
                    if (e <= levelLo || e <= 1e-8f)
                        continue;
                    // multi-scale isolation: a firefly seen through the pixel
                    // filter is a small blob, so we also test rings that start
                    // outside its footprint and keep the best scale.
                    float bestD = -1e30f, bestHi = 0.f;
                    int bestS = 1;
                    for (int s = 1; s <= p.size; ++s)
                    {
                        for (int q = 0; q < kRankHigh; ++q)
                            top[q] = 0.f;
                        int cnt = 0;
                        for (const auto& o : cx.rings[s])
                        {
                            const int xx = x + o.first, yy = y + o.second;
                            if (xx < 0 || yy < 0 || xx >= W || yy >= H)
                                continue;
                            const float v = En[yy * W + xx];
                            ++cnt;
                            if (v > top[kRankHigh - 1])
                            {
                                int q = kRankHigh - 1;
                                while (q > 0 && top[q - 1] < v)
                                {
                                    top[q] = top[q - 1];
                                    --q;
                                }
                                top[q] = v;
                            }
                        }
                        const float hi = top[std::min(kRankHigh, std::max(cnt, 1)) - 1];
                        const float d = stops(e, hi);
                        if (d > bestD)
                        {
                            bestD = d;
                            bestHi = hi;
                            bestS = s;
                        }
                        if (bestD >= p.isolation + p.softness)
                            break; // already fully isolated
                    }
                    if (bestD <= p.isolation)
                        continue;
                    list.push_back({i, bestS, bestD, bestHi, 0.f, 0.f, -1, 0.f});
                }
            });
        }
        for (auto& th : pool)
            th.join();
        std::vector<Cand> res;
        for (auto& l : perThread)
            res.insert(res.end(), l.begin(), l.end());
        return res;
    };

    // Local statistics + statistical matte for candidates [from, end).
    auto statsPass = [&](std::vector<Cand>& cs, size_t from, const float* En, const std::vector<uint8_t>& candMap) {
        parallelFor((int)(cs.size() - from), nth, [&](int b, int e) {
            float vals[64];
            for (int k = b; k < e; ++k)
            {
                Cand& c = cs[from + k];
                const int x = c.idx % W, y = c.idx / W;
                int n = 0;
                for (const auto& o : cx.rings[c.scale])
                {
                    const int xx = x + o.first, yy = y + o.second;
                    if (xx < 0 || yy < 0 || xx >= W || yy >= H)
                        continue;
                    vals[n++] = En[yy * W + xx];
                }
                float med = 0.f;
                if (n > 0)
                {
                    std::nth_element(vals, vals + n / 2, vals + n);
                    med = vals[n / 2];
                }
                c.med = cx.medoid(c.idx, candMap, c.scale, En);
                c.eref = c.med >= 0 ? En[c.med] : med;
                if (cx.bad[c.idx])
                    continue; // forced
                c.S = std::max(0.f, stops(c.hi, med));
                c.m = matteOf(En[c.idx], c.eref, c.S, c.diso);
            }
        });
    };

    std::vector<uint8_t> candMap(N, 0);
    std::vector<Cand> cands = isolationPass(nullptr, cx.E.data(), candMap);
    for (const auto& c : cands)
        candMap[c.idx] = 1;
    statsPass(cands, 0, cx.E.data(), candMap);
    if (isAborted())
        return false;

    // Peeling: fireflies often come in clusters, and each one hides the others
    // from the isolation test. Flatten what was found and look again around it
    // (including weak candidates, which may now turn out to be isolated).
    {
        std::vector<float> Ew;
        std::vector<int> candPos(N, -1);
        for (size_t k = 0; k < cands.size(); ++k)
            candPos[cands[k].idx] = (int)k;
        std::vector<int> fresh(cands.size());
        for (size_t k = 0; k < cands.size(); ++k)
            fresh[k] = (int)k;
        for (int iter = 0; iter < 2 && !fresh.empty(); ++iter)
        {
            std::vector<int> peeled;
            for (int k : fresh)
                if (cands[k].m > 0.5f || cx.bad[cands[k].idx])
                    peeled.push_back(k);
            if (peeled.empty())
                break;
            if (Ew.empty())
                Ew = cx.E;
            std::vector<uint8_t> strong(N, 0), near(N, 0);
            for (const auto& cd : cands)
                if (cd.m > 0.5f || cx.bad[cd.idx])
                    strong[cd.idx] = 1;
            const int r = p.size + 1 + p.grow;
            for (int k : peeled)
            {
                const Cand& cd = cands[k];
                Ew[cd.idx] = std::min(Ew[cd.idx], cd.eref);
                const int x = cd.idx % W, y = cd.idx / W;
                for (int dy = -r; dy <= r; ++dy)
                    for (int dx = -r; dx <= r; ++dx)
                    {
                        const int xx = x + dx, yy = y + dy;
                        if (xx < 0 || yy < 0 || xx >= W || yy >= H)
                            continue;
                        const int q = yy * W + xx;
                        near[q] = 1;
                    }
            }
            std::vector<int> subset;
            for (int i = 0; i < N; ++i)
                if (near[i] && !strong[i])
                    subset.push_back(i);
            std::vector<Cand> more = isolationPass(&subset, Ew.data(), strong);
            if (more.empty())
                break;
            for (const auto& cd : more)
                candMap[cd.idx] = 1;
            statsPass(more, 0, Ew.data(), candMap);
            fresh.clear();
            for (const auto& cd : more)
            {
                const int pos = candPos[cd.idx];
                if (pos < 0)
                {
                    candPos[cd.idx] = (int)cands.size();
                    fresh.push_back((int)cands.size());
                    cands.push_back(cd);
                }
                else if (cd.m > cands[pos].m)
                {
                    cands[pos] = cd;
                    fresh.push_back(pos);
                }
            }
            if (isAborted())
                return false;
        }
    }

    // -- OIDN expected value -------------------------------------------------
    std::vector<float> ref; // interleaved RGB, empty if statistical
    if (p.reference == ReferenceMode::OIDN)
    {
        if (!reference)
            out.warning = "OIDN unavailable - used the statistical reference";
        else
        {
            // Pre-clean the frame so that the fireflies do not leak into the
            // prediction (OIDN would otherwise turn them into soft blobs).
            std::vector<float> clean((size_t)N * 3);
            parallelFor(H, nth, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < W; ++x)
                    {
                        const size_t i = (size_t)y * W + x;
                        for (int c = 0; c < 3; ++c)
                            clean[i * 3 + c] = std::max(0.f, finiteOr(in.beauty.at(x, y, c), 0.f));
                    }
            });
            const float relax = p.preCleanBias; // clean more aggressively than we correct
            for (const auto& c : cands)
            {
                float m = c.m;
                if (!cx.bad[c.idx])
                {
                    const float teff = p.threshold - relax + p.adaptivity * c.S;
                    m = smoothstep(teff, teff + p.softness, stops(cx.E[c.idx], c.eref)) *
                        smoothstep(p.isolation, p.isolation + p.softness, c.diso);
                }
                else
                    m = 1.f;
                if (m <= 0.f)
                    continue;
                const size_t i = (size_t)c.idx;
                for (int ch = 0; ch < 3; ++ch)
                {
                    const float target = c.med >= 0 ? std::max(0.f, cx.px(c.med, ch)) : 0.f;
                    clean[i * 3 + ch] += m * (target - clean[i * 3 + ch]);
                }
                // flatten the pixel-filter halo around it to the local level
                if (m > 0.5f && p.grow > 0)
                {
                    const int x = c.idx % W, y = c.idx / W, g = p.grow;
                    const float cap = 1.25f * c.eref + kEps;
                    for (int dy = -g; dy <= g; ++dy)
                        for (int dx = -g; dx <= g; ++dx)
                        {
                            const int xx = x + dx, yy = y + dy;
                            if ((dx == 0 && dy == 0) || xx < 0 || yy < 0 || xx >= W || yy >= H)
                                continue;
                            const size_t q = (size_t)yy * W + xx;
                            const float eq = energyOf(clean[q * 3], clean[q * 3 + 1], clean[q * 3 + 2]);
                            if (eq > cap)
                                for (int ch = 0; ch < 3; ++ch)
                                    clean[q * 3 + ch] *= cap / eq;
                        }
                }
            }
            if (isAborted())
                return false;
            ref.assign((size_t)N * 3, 0.f);
            std::string rerr;
            if (!reference(W, H, clean.data(), in.albedo, in.normal, ref.data(), rerr))
            {
                if (isAborted())
                    return false;
                out.warning = "OIDN failed (" + rerr + ") - used the statistical reference";
                ref.clear();
            }
        }
        if (isAborted())
            return false;
    }
    const bool useRef = !ref.empty();
    auto refEnergy = [&](int idx) {
        const float* r = &ref[(size_t)idx * 3];
        return energyOf(r[0], r[1], r[2]);
    };

    // -- final matte on candidates ------------------------------------------
    parallelFor((int)cands.size(), nth, [&](int b, int e) {
        float top[kRankHigh];
        for (int k = b; k < e; ++k)
        {
            Cand& c = cands[k];
            if (cx.bad[c.idx])
            {
                c.m = 1.f;
            }
            else
            {
                if (useRef)
                {
                    c.m = matteOf(cx.E[c.idx], refEnergy(c.idx), c.S, c.diso);
                }
                if (p.temporal && c.m > 0.f && (in.prev.valid() || in.next.valid()))
                {
                    const int x = c.idx % W, y = c.idx / W;
                    float th = 0.f;
                    if (in.prev.valid())
                        th = std::max(th, windowHigh(in.prev, W, H, x, y, p.motionRadius, top));
                    if (in.next.valid())
                        th = std::max(th, windowHigh(in.next, W, H, x, y, p.motionRadius, top));
                    c.m *= smoothstep(p.isolation, p.isolation + p.softness, stops(cx.E[c.idx], th));
                }
            }
            if (in.mask)
            {
                const int x = c.idx % W, y = c.idx / W;
                float mk = in.mask[(int64_t)y * in.maskRowStride + x];
                mk = std::isfinite(mk) ? std::max(0.f, std::min(1.f, mk)) : 0.f;
                c.m *= mk;
            }
        }
    });
    if (isAborted())
        return false;

    // -- hits: core + halo ---------------------------------------------------
    out.matte.assign(N, 0.f);
    std::vector<uint8_t> hitMap(N, 0);
    std::vector<int> coreIdx;
    std::vector<float> coreRef; // expected energy, used by the halo test
    for (const auto& c : cands)
        if (c.m > kMinHit)
        {
            out.matte[c.idx] = c.m;
            hitMap[c.idx] = 1;
            coreIdx.push_back(c.idx);
            coreRef.push_back(useRef ? refEnergy(c.idx) : c.eref);
        }
    out.numCore = (int)coreIdx.size();

    std::vector<int> haloIdx;
    if (p.grow > 0)
    {
        const int g = p.grow;
        for (size_t k = 0; k < coreIdx.size(); ++k)
        {
            const int idx = coreIdx[k], x = idx % W, y = idx / W;
            const float mp = out.matte[idx];
            for (int dy = -g; dy <= g; ++dy)
                for (int dx = -g; dx <= g; ++dx)
                {
                    const int xx = x + dx, yy = y + dy;
                    if (xx < 0 || yy < 0 || xx >= W || yy >= H)
                        continue;
                    const int q = yy * W + xx;
                    if (cx.bad[q] || q == idx)
                        continue;
                    // halo pixel: corrected only if it is itself above the local
                    // background (a real neighbouring sample, the firefly excluded).
                    // The OIDN prediction is not used here: right next to a firefly it
                    // is still slightly lifted by it.
                    // Tested per channel: a coloured firefly on a coloured surface
                    // leaves a halo that only shows in the "other" channels.
                    const int mo = cx.medoid(q, hitMap, p.size);
                    float dmax = stops(cx.E[q], mo >= 0 ? cx.E[mo] : coreRef[k]);
                    if (mo >= 0)
                    {
                        const float eq = cx.E[q] + kEps;
                        for (int ch = 0; ch < 3; ++ch)
                        {
                            const float v = std::max(0.f, cx.px(q, ch)), r = std::max(0.f, cx.px(mo, ch));
                            // ignore channels that are a negligible part of the pixel
                            if (v > 0.05f * eq)
                                dmax = std::max(dmax, stops(v, r) - 0.25f);
                        }
                    }
                    const float hw = smoothstep(0.2f, 1.0f, dmax);
                    float m = mp * hw;
                    if (in.mask)
                    {
                        float mk = in.mask[(int64_t)yy * in.maskRowStride + xx];
                        m *= std::isfinite(mk) ? std::max(0.f, std::min(1.f, mk)) : 0.f;
                    }
                    if (m <= kMinHit)
                        continue;
                    if (hitMap[q] == 0)
                    {
                        hitMap[q] = 2;
                        haloIdx.push_back(q);
                    }
                    out.matte[q] = std::max(out.matte[q], m);
                }
        }
    }

    std::vector<int>& hits = out.hitIndex;
    hits = coreIdx;
    hits.insert(hits.end(), haloIdx.begin(), haloIdx.end());
    const int nh = (int)hits.size();
    out.hitM.resize(nh);
    out.hitDelta.assign((size_t)nh * 3, 0.f);
    out.hitValue.assign((size_t)nh * 3, 0.f);
    out.hitIn.assign((size_t)nh * 3, 0.f);
    out.hitMed.assign((size_t)nh * 3, 0.f);
    out.hitFlags.assign(nh, 0);
    if (isAborted())
        return false;

    // -- corrected values ----------------------------------------------------
    parallelFor(nh, nth, [&](int b, int e) {
        for (int k = b; k < e; ++k)
        {
            const int idx = hits[k];
            const int x = idx % W, y = idx / W;
            const float m = out.matte[idx];
            out.hitM[k] = m;
            uint8_t flags = (k < out.numCore) ? 2 : 0;

            float med[3];
            float inV[3] = {in.beauty.at(x, y, 0), in.beauty.at(x, y, 1), in.beauty.at(x, y, 2)};
            const bool okMed = cx.ringMedian(idx, hitMap, p.size, med);
            const int mo = cx.medoid(idx, hitMap, p.size);
            // core hits use the expected value (OIDN or statistical); halo hits
            // always use the local background sample (see the halo test above)
            const bool useOidnHere = useRef && (k < out.numCore);
            float refC[3];
            for (int c = 0; c < 3; ++c)
                refC[c] = std::max(0.f, useOidnHere ? ref[(size_t)idx * 3 + c]
                                                    : (mo >= 0 ? finiteOr(cx.px(mo, c), 0.f) : (okMed ? med[c] : 0.f)));

            float V[3];
            switch (p.method)
            {
            case Method::LocalMedian:
                for (int c = 0; c < 3; ++c)
                    V[c] = okMed ? med[c] : refC[c];
                break;
            case Method::Clamp:
            {
                const float e0 = energyOf(finiteOr(inV[0], 0.f), finiteOr(inV[1], 0.f), finiteOr(inV[2], 0.f));
                const float level = energyOf(refC[0], refC[1], refC[2]) * std::exp2(p.threshold);
                const float s = e0 > level ? level / e0 : 1.f;
                for (int c = 0; c < 3; ++c)
                    V[c] = std::isfinite(inV[c]) ? inV[c] * s : refC[c];
                break;
            }
            case Method::Reference:
            default:
                for (int c = 0; c < 3; ++c)
                    V[c] = refC[c];
                break;
            }

            for (int c = 0; c < 3; ++c)
            {
                if (!std::isfinite(inV[c]))
                {
                    flags |= 1;
                    inV[c] = V[c];
                }
                out.hitIn[(size_t)k * 3 + c] = inV[c];
                out.hitValue[(size_t)k * 3 + c] = V[c];
                out.hitMed[(size_t)k * 3 + c] = okMed ? med[c] : V[c];
                out.hitDelta[(size_t)k * 3 + c] = m * (inV[c] - V[c]);
            }
            out.hitFlags[k] = flags;
        }
    });

    // -- energy kept (see compensateLayer): share of each hit's removed energy
    // that is given back. Only where the detections are dense AND carry a real
    // part of the local light: then they are the noise of an under-sampled
    // light, not accidents (elsewhere, giving them back would only add noise).
    // Each value is also capped: the samples of an under-sampled light reach
    // ~1000x the local level; beyond that, a value is an accident that must
    // not turn into a glow.
    out.hitKeep.assign(nh, 0.f);
    if (p.conserveEnergy && p.energyRadius > 0.f && nh > 0)
    {
        const lp::Grid g = lp::grid(W, H, p.energyRadius);
        const size_t nc = (size_t)g.cw * g.ch;
        std::vector<float> lvl, ones, rem(nc, 0.f), cnt(nc, 0.f);
        lp::down(g, [&](int x, int y) { return cx.E[y * W + x]; }, lvl, nth);
        std::vector<float> dE(nh);
        for (int k = 0; k < nh; ++k)
        {
            const int idx = hits[k], x = idx % W, y = idx / W;
            const size_t cell = (size_t)(y / g.f) * g.cw + x / g.f;
            const float d = std::min(cx.E[idx], energyOf(out.hitDelta[(size_t)k * 3], out.hitDelta[(size_t)k * 3 + 1],
                                                         out.hitDelta[(size_t)k * 3 + 2]));
            dE[k] = d;
            if (out.hitFlags[k] & 1)
                continue;
            lvl[cell] -= d;
            rem[cell] += d;
            if (out.hitFlags[k] & 2)
                cnt[cell] += out.hitM[k];
        }
        lp::down(g, [](int, int) { return 1.f; }, ones, nth);
        for (std::vector<float>* v : {&lvl, &ones, &rem, &cnt})
            lp::blur(g, *v, nth);
        const float area = 2.f * 3.14159265f * p.energyRadius * p.energyRadius;
        parallelFor(nh, nth, [&](int b, int e) {
            for (int k = b; k < e; ++k)
            {
                if (out.hitFlags[k] & 1)
                    continue; // NaN / inf: no energy to give back
                const int idx = hits[k];
                const float fx = (float)(idx % W), fy = (float)(idx / W);
                const float o1 = lp::at(g, ones, fx, fy);
                if (o1 <= 0.f)
                    continue;
                const float level = std::max(0.f, lp::at(g, lvl, fx, fy) / o1);
                const float removed = std::max(0.f, lp::at(g, rem, fx, fy) / o1);
                const float dense = lp::smoothstep(2.f, 6.f, lp::at(g, cnt, fx, fy) / o1 * area);
                const float share = lp::smoothstep(0.03f, 0.1f, removed / std::max(1e-12f, removed + level));
                const float cap = 1024.f * level;
                out.hitKeep[k] = dense * share * (dE[k] > cap ? cap / dE[k] : 1.f);
            }
        });
    }

    out.valid = true;
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

// ===========================================================================
// AOVs
// ===========================================================================

void beginAovs(Analysis& a, int numLayers)
{
    a.aovs.assign(std::max(0, numLayers), Analysis::Aov());
    const size_t n = a.hitIndex.size() * 3;
    for (auto& l : a.aovs)
    {
        l.delta.assign(n, 0.f);
        l.med.assign(n, 0.f);
        l.val.assign(n, 0.f);
        l.excess.assign(n, 0.f);
    }
    a.aovsFinalized = false;
}

void addAovChannel(Analysis& a, const Params& pIn, int layer, int comp, const float* data, int64_t rowStride)
{
    if (layer < 0 || layer >= (int)a.aovs.size() || comp < 0 || comp > 2 || !data)
        return;
    Analysis::Aov& L = a.aovs[layer];
    const int W = a.width, H = a.height;
    const auto ring = ringOffsets(std::max(1, std::min(3, pIn.size)));
    const int nh = (int)a.hitIndex.size();
    parallelFor(nh, threadCount(pIn), [&](int b, int e) {
        float v[64];
        for (int k = b; k < e; ++k)
        {
            const int idx = a.hitIndex[k], x = idx % W, y = idx / W;
            int n = 0;
            for (const auto& o : ring)
            {
                const int xx = x + o.first, yy = y + o.second;
                if (xx < 0 || yy < 0 || xx >= W || yy >= H)
                    continue;
                if (a.matte[(size_t)yy * W + xx] > 0.f)
                    continue;
                const float f = data[(int64_t)yy * rowStride + xx];
                if (std::isfinite(f))
                    v[n++] = f;
            }
            float med = 0.f;
            if (n > 0)
            {
                std::nth_element(v, v + n / 2, v + n);
                med = v[n / 2];
            }
            const float val = data[(int64_t)y * rowStride + x];
            const size_t o = (size_t)k * 3 + comp;
            L.med[o] = med;
            if (std::isfinite(val))
            {
                L.val[o] = val;
                L.excess[o] = std::max(0.f, val - med);
            }
            else
            {
                L.val[o] = med;
                L.excess[o] = 0.f;
            }
        }
    });
}

void finalizeAovs(Analysis& a)
{
    const size_t nh = a.hitIndex.size();
    for (size_t k = 0; k < nh; ++k)
        for (int c = 0; c < 3; ++c)
        {
            const size_t o = k * 3 + c;
            const float db = a.hitDelta[o];
            if (db == 0.f)
                continue;
            float sumE = 0.f, sumV = 0.f;
            for (const auto& L : a.aovs)
            {
                sumE += L.excess[o];
                sumV += std::max(0.f, L.val[o]);
            }
            const float bin = a.hitIn[o];
            const float unselected = std::max(0.f, bin - sumV);            // beauty not covered by the AOVs
            const float beautyExcess = std::max(0.f, bin - a.hitMed[o]);
            const float eUnsel = std::min(unselected, std::max(0.f, beautyExcess - sumE));
            const float denomE = sumE + eUnsel;
            const float denomV = sumV + unselected;
            if (db > 0.f)
            {
                // 1) take the removed energy from the AOVs that carry the excess
                const float share = denomE > 1e-8f ? std::min(db, denomE) : 0.f;
                float capSum = 0.f;
                for (auto& L : a.aovs)
                {
                    L.delta[o] = share > 0.f ? share * L.excess[o] / denomE : 0.f; // <= excess <= value
                    capSum += std::max(0.f, L.val[o] - L.delta[o]);
                }
                // 2) anything left is taken in proportion to what each AOV still holds
                //    (the part of the beauty outside the selected AOVs takes its share too)
                const float rem = db - share;
                const float unselCap = std::max(0.f, unselected - (share - std::min(share, sumE)));
                const float denomC = capSum + unselCap;
                if (rem > 0.f && denomC > 1e-8f)
                {
                    const float s = std::min(1.f, rem / denomC);
                    for (auto& L : a.aovs)
                        L.delta[o] += s * std::max(0.f, L.val[o] - L.delta[o]);
                }
            }
            else if (denomV > 1e-8f)
            {
                const float s = db / denomV;
                for (auto& L : a.aovs)
                    L.delta[o] = std::max(0.f, L.val[o]) * s;
            }
        }
    for (auto& L : a.aovs)
    {
        std::vector<float>().swap(L.val);
        std::vector<float>().swap(L.excess);
    }
    a.aovsFinalized = true;
}

// ===========================================================================
// apply
// ===========================================================================

void applyLayer(const Analysis& a, const Params& /*p*/, int layer, const OutBuffer& ob, float mix)
{
    if (!a.valid || mix <= 0.f)
        return;
    const std::vector<float>* delta = &a.hitDelta;
    const std::vector<float>* fallback = &a.hitValue;
    if (layer >= 0)
    {
        if (layer >= (int)a.aovs.size() || !a.aovsFinalized)
            return;
        delta = &a.aovs[layer].delta;
        fallback = &a.aovs[layer].med;
    }
    const int W = a.width;
    const int nh = (int)a.hitIndex.size();

    auto inBuf = [&](int x, int y) {
        const int bx = x - ob.x0, by = y - ob.y0;
        return bx >= 0 && by >= 0 && bx < ob.width && by < ob.height;
    };
    auto ptr = [&](int c, int x, int y) { return ob.ch[c] + (int64_t)(y - ob.y0) * ob.rowStride + (int64_t)(x - ob.x0) * ob.pixStride; };

    for (int k = 0; k < nh; ++k)
    {
        const int idx = a.hitIndex[k], x = idx % W, y = idx / W;
        if (!inBuf(x, y))
            continue;
        for (int c = 0; c < 3; ++c)
        {
            if (!ob.ch[c])
                continue;
            float* v = ptr(c, x, y);
            if (!std::isfinite(*v))
            {
                if (a.hitM[k] > 0.f)
                    *v = (*fallback)[(size_t)k * 3 + c];
                continue;
            }
            const float before = *v;
            *v -= mix * (*delta)[(size_t)k * 3 + c];
            if (before >= 0.f && *v < 0.f)
                *v = 0.f; // float rounding: a positive pixel never turns negative
        }
    }
}

void compensateLayer(const Analysis& a, const Params& p, int layer, const OutBuffer& ob, const View3& albedo,
                     float mix, int threads)
{
    const float amount = p.energyAmount * mix;
    if (!a.valid || !p.conserveEnergy || amount <= 0.f || p.energyRadius <= 0.f || a.hitIndex.empty() ||
        a.hitKeep.size() != a.hitIndex.size())
        return;
    const std::vector<float>* delta = &a.hitDelta;
    if (layer >= 0)
    {
        if (layer >= (int)a.aovs.size() || !a.aovsFinalized)
            return;
        delta = &a.aovs[layer].delta;
    }
    const int W = a.width, H = a.height;
    const int nth = detail::threadCount(threads);
    const lp::Grid g = lp::grid(W, H, p.energyRadius);
    const size_t nc = (size_t)g.cw * g.ch;

    // removed energy that is given back, per cell (hits are few: done serially)
    std::vector<float> rem[3];
    for (int c = 0; c < 3; ++c)
        rem[c].assign(nc, 0.f);
    bool any = false;
    for (size_t k = 0; k < a.hitIndex.size(); ++k)
    {
        const float keep = a.hitKeep[k];
        if (keep <= 0.f)
            continue;
        const int idx = a.hitIndex[k], x = idx % W, y = idx / W;
        const size_t cell = (size_t)(y / g.f) * g.cw + x / g.f;
        for (int c = 0; c < 3; ++c)
        {
            const float d = keep * (*delta)[k * 3 + c];
            if (std::isfinite(d) && d != 0.f)
            {
                rem[c][cell] += d;
                any = true;
            }
        }
    }
    if (!any)
        return;
    for (int c = 0; c < 3; ++c)
        lp::blur(g, rem[c], nth);

    // where the energy goes back: in proportion to the albedo (indirect light
    // is the albedo times a smooth irradiance)
    auto alb = [&](int x, int y, int c) {
        if (!albedo.valid())
            return 1.f;
        const float v = albedo.at(x, y, c);
        return std::isfinite(v) ? std::max(0.02f, std::min(1.f, v)) : 0.5f;
    };
    std::vector<float> ga[3];
    for (int c = 0; c < 3; ++c)
    {
        lp::down(g, [&](int x, int y) { return alb(x, y, c); }, ga[c], nth);
        lp::blur(g, ga[c], nth);
    }
    detail::parallelFor(ob.height, nth, [&](int y0, int y1) {
        for (int by = y0; by < y1; ++by)
        {
            const int y = by + ob.y0;
            if (y < 0 || y >= H)
                continue;
            for (int bx = 0; bx < ob.width; ++bx)
            {
                const int x = bx + ob.x0;
                if (x < 0 || x >= W)
                    continue;
                for (int c = 0; c < 3; ++c)
                {
                    if (!ob.ch[c])
                        continue;
                    float* v = ob.ch[c] + (int64_t)by * ob.rowStride + (int64_t)bx * ob.pixStride;
                    if (!std::isfinite(*v))
                        continue;
                    const float r = lp::at(g, rem[c], (float)x, (float)y), da = lp::at(g, ga[c], (float)x, (float)y);
                    if (r != 0.f && da > 1e-12f)
                        *v += amount * r * alb(x, y, c) / da;
                }
            }
        }
    });
}

void writeMatte(const Analysis& a, const OutBuffer& ob, float mix)
{
    if (!a.valid || !ob.ch[0])
        return;
    const int W = a.width;
    for (size_t k = 0; k < a.hitIndex.size(); ++k)
    {
        const int idx = a.hitIndex[k], x = idx % W - ob.x0, y = idx / W - ob.y0;
        if (x < 0 || y < 0 || x >= ob.width || y >= ob.height)
            continue;
        ob.ch[0][(int64_t)y * ob.rowStride + (int64_t)x * ob.pixStride] = a.hitM[k] * mix;
    }
}

} // namespace cleanpass
