// CleanPass - the three cleaning stages chained together
// Copyright (c) 2026 Gabin Vidal. MIT License (see LICENSE).

#include "cp_pipeline.h"
#include "cp_parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace cleanpass
{
namespace
{

double secondsSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Copies one channel of a view into an x-contiguous buffer (w*h).
void toPlanar(const View3& v, int c, int W, int H, std::vector<float>& dst)
{
    dst.resize((size_t)W * H);
    detail::parallelFor(H, detail::threadCount(0), [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
                dst[(size_t)y * W + x] = v.at(x, y, c);
    });
}

inline float tmap(float v) { return std::isfinite(v) && v > 0.f ? v / (1.f + v) : 0.f; }

inline float bilinear(const float* img, int W, int H, float fx, float fy)
{
    fx = std::max(0.f, std::min((float)(W - 1), fx));
    fy = std::max(0.f, std::min((float)(H - 1), fy));
    const int x0 = std::min(std::max(0, W - 2), (int)fx), y0 = std::min(std::max(0, H - 2), (int)fy);
    const int x1 = std::min(W - 1, x0 + 1), y1 = std::min(H - 1, y0 + 1);
    const float tx = fx - x0, ty = fy - y0;
    const float a = img[(size_t)y0 * W + x0], b = img[(size_t)y0 * W + x1];
    const float c = img[(size_t)y1 * W + x0], d = img[(size_t)y1 * W + x1];
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

// OIDN on a planar 3-channel image, mixed with `amount`. Returns false on failure.
bool denoisePlanar(const ReferenceFn& denoise, int W, int H, const View3& alb, const View3& nrm, float amount,
                   std::vector<float>& img, std::string& err)
{
    const size_t N = (size_t)W * H;
    const int nth = detail::threadCount(0);
    std::vector<float> color(N * 3), den(N * 3, 0.f);
    detail::parallelFor(H, nth, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                const size_t i = (size_t)y * W + x;
                for (int c = 0; c < 3; ++c)
                {
                    const float v = img[c * N + i];
                    color[i * 3 + c] = std::isfinite(v) ? std::max(0.f, v) : 0.f;
                }
            }
    });
    if (!denoise(W, H, color.data(), alb, nrm, den.data(), err))
        return false;
    const float a = std::max(0.f, std::min(1.f, amount));
    detail::parallelFor(H, nth, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                const size_t i = (size_t)y * W + x;
                for (int c = 0; c < 3; ++c)
                {
                    float& v = img[c * N + i];
                    const float d = den[i * 3 + c];
                    if (std::isfinite(d))
                        v = std::isfinite(v) ? v + a * (d - v) : d;
                }
            }
    });
    return true;
}

} // namespace

bool runPipeline(const PipelineInput& in, const PipelineParams& p, const ReferenceFn& reference,
                 const ReferenceFn& denoise, PipelineResult& out, std::string& err, const AbortFn& aborted)
{
    out = PipelineResult();
    const int W = in.width, H = in.height;
    out.width = W;
    out.height = H;
    if (W <= 0 || H <= 0 || !in.cur.beauty.valid())
    {
        err = "no input image";
        return false;
    }
    const size_t N = (size_t)W * H;
    auto isAborted = [&]() { return aborted && aborted(); };
    const int nth = detail::threadCount(0);
    const bool wantStab = p.denoise && p.stabilize && p.denoiseAmount > 0.f && denoise;

    // A frame already processed by the same chain (stages 1 - 3): reuse it.
    const PipelineResult* pre = in.precomputed;
    if (pre && (!pre->valid || pre->stabilized || pre->width != W || pre->height != H || pre->beauty.size() != N * 3 ||
                (p.denoise && pre->beautyStage2.size() != N * 3) || (p.deflicker && p.df.frames > 1)))
        pre = nullptr;
    // its deflicker neighbours must all be here (else the AOVs could not follow)
    std::vector<int> preNeighbour;
    if (pre && pre->deflicker.valid)
        for (int o : pre->dfOffset)
        {
            int idx = -1;
            for (size_t i = 0; i < in.neighbours.size(); ++i)
                if (in.neighbours[i].offset == o && in.neighbours[i].beauty.valid())
                    idx = (int)i;
            if (idx < 0)
            {
                pre = nullptr;
                break;
            }
            preNeighbour.push_back(idx);
        }
    auto t0 = std::chrono::steady_clock::now();
    bool denoised = false;
    if (pre)
    {
        out.firefly = pre->firefly;
        out.beauty = pre->beauty;
        out.beautyStage2 = pre->beautyStage2;
        out.deflicker = pre->deflicker;
        out.dfOffset = pre->dfOffset;
        out.dfNeighbour = preNeighbour;
        out.reused = true;
        denoised = p.denoise && !out.beautyStage2.empty();
    }
    else
    {
        // ---- 1. Firefly ------------------------------------------------------
        if (p.firefly)
        {
            Inputs fi;
            fi.width = W;
            fi.height = H;
            fi.beauty = in.cur.beauty;
            fi.albedo = in.cur.albedo;
            fi.normal = in.cur.normal;
            if (p.ff.temporal)
                for (const auto& n : in.neighbours)
                {
                    if (n.offset == -1 && n.beauty.valid())
                        fi.prev = n.beauty;
                    if (n.offset == 1 && n.beauty.valid())
                        fi.next = n.beauty;
                }
            fi.mask = in.mask;
            fi.maskRowStride = in.maskStride;
            if (!analyze(fi, p.ff, reference, out.firefly, err, aborted))
                return false;
            if (!out.firefly.warning.empty())
                out.warning = out.firefly.warning;
        }
        else
        {
            out.firefly.width = W;
            out.firefly.height = H;
            out.firefly.matte.assign(N, 0.f);
            out.firefly.valid = true;
        }
        out.seconds[0] = secondsSince(t0);

        // beauty after stage 1 (planar)
        out.beauty.resize(N * 3);
        detail::parallelFor(H, nth, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x)
                    for (int c = 0; c < 3; ++c)
                        out.beauty[c * N + (size_t)y * W + x] = in.cur.beauty.at(x, y, c);
        });
        {
            OutBuffer ob;
            for (int c = 0; c < 3; ++c)
                ob.ch[c] = out.beauty.data() + c * N;
            ob.width = W;
            ob.height = H;
            ob.rowStride = W;
            applyLayer(out.firefly, p.ff, -1, ob, 1.f);
            compensateLayer(out.firefly, p.ff, -1, ob, in.cur.albedo, 1.f);
        }
        if (isAborted())
            return false;

        // ---- 2. Deflicker (also run, analysis only, for the stabilisation) -----
        t0 = std::chrono::steady_clock::now();
        if ((p.deflicker || wantStab || p.keepAnalysis) && !in.neighbours.empty())
        {
            DeflickerParams dp = p.df;
            if (!p.deflicker)
                dp.frames = 1;
            DeflickerFrame cur;
            cur.beauty = View3::planar(out.beauty.data(), out.beauty.data() + N, out.beauty.data() + 2 * N, W);
            cur.albedo = in.cur.albedo;
            cur.normal = in.cur.normal;
            cur.depth = in.cur.depth;
            cur.motion = in.cur.motion;
            cur.variance = in.cur.variance;
            cur.raw = in.cur.beauty;
            std::vector<DeflickerFrame> nbs;
            for (size_t i = 0; i < in.neighbours.size(); ++i)
            {
                const auto& n = in.neighbours[i];
                if (!n.beauty.valid() || n.offset == 0 || std::abs(n.offset) > std::max(1, dp.frames))
                    continue;
                DeflickerFrame f;
                f.beauty = n.beauty;
                f.albedo = n.albedo;
                f.normal = n.normal;
                f.depth = n.depth;
                f.offset = n.offset;
                nbs.push_back(f);
                out.dfNeighbour.push_back((int)i);
                out.dfOffset.push_back(n.offset);
            }
            if (!deflickerAnalyze(W, H, cur, nbs, dp, out.deflicker, aborted))
                return false;
            if (p.deflicker)
            {
                // apply, channel by channel
                std::vector<float> curC, result(N);
                std::vector<std::vector<float>> nbC(nbs.size());
                for (int c = 0; c < 3; ++c)
                {
                    curC.assign(out.beauty.begin() + c * N, out.beauty.begin() + (c + 1) * N);
                    std::vector<const float*> ptr;
                    std::vector<int64_t> st;
                    for (size_t k = 0; k < nbs.size(); ++k)
                    {
                        toPlanar(nbs[k].beauty, c, W, H, nbC[k]);
                        ptr.push_back(nbC[k].data());
                        st.push_back(W);
                    }
                    deflickerApplyChannel(out.deflicker, dp.amount, curC.data(), W, ptr, st, result.data(), W);
                    std::memcpy(out.beauty.data() + c * N, result.data(), N * sizeof(float));
                }
            }
        }
        out.seconds[1] = secondsSince(t0);
        if (isAborted())
            return false;

        // ---- 3. Denoise ------------------------------------------------------
        t0 = std::chrono::steady_clock::now();
        if (p.denoise && p.denoiseAmount > 0.f)
        {
            if (!denoise)
                out.warning = "OIDN unavailable - denoise skipped";
            else
            {
                out.beautyStage2 = out.beauty;
                std::string derr;
                if (!denoisePlanar(denoise, W, H, in.cur.albedo, in.cur.normal, p.denoiseAmount, out.beauty, derr))
                {
                    if (isAborted())
                        return false;
                    out.warning = "denoise failed (" + derr + ")";
                    out.beautyStage2.clear();
                }
                else
                    denoised = true;
            }
        }

    } // stages 1 - 3

    // ---- 4. Stabilisation of the denoised result ---------------------------
    if (denoised && wantStab && out.deflicker.valid && out.deflicker.count > 0)
    {
        out.beautyDenoised = out.beauty;
        std::vector<float> acc(out.beauty), wsum(N, 1.f);
        for (int o = -1; o <= 1; o += 2)
        {
            // deflicker entry of this neighbour (its motion and geometric weight)
            int k = -1;
            for (int q = 0; q < out.deflicker.count && q < (int)out.dfNeighbour.size(); ++q)
                if (in.neighbours[out.dfNeighbour[q]].offset == o)
                    k = q;
            if (k < 0)
                continue;
            // denoised result of that frame: given, or computed with the same chain
            const std::vector<float>* given = o < 0 ? in.denoisedPrev : in.denoisedNext;
            std::shared_ptr<PipelineResult>& mine = o < 0 ? out.subPrev : out.subNext;
            const std::vector<float>* Dk = (given && given->size() == N * 3) ? given : nullptr;
            if (!Dk)
            {
                const PipelineFrame& nf = in.neighbours[out.dfNeighbour[k]];
                PipelineInput sub;
                sub.width = W;
                sub.height = H;
                sub.cur = nf;
                sub.cur.offset = 0;
                sub.mask = in.mask;
                sub.maskStride = in.maskStride;
                auto addRel = [&](const PipelineFrame& f) {
                    const int rel = f.offset - o;
                    if (std::abs(rel) == 1 && f.beauty.valid())
                    {
                        PipelineFrame g = f;
                        g.offset = rel;
                        g.motion = View3();
                        g.variance = View3();
                        sub.neighbours.push_back(g);
                    }
                };
                addRel(in.cur);
                for (const auto& f : in.neighbours)
                    addRel(f);
                PipelineParams sp = p;
                sp.stabilize = false;
                sp.df.frames = 1;
                sp.keepAnalysis = true; // so that the frame can reuse it when its turn comes
                auto sr = std::make_shared<PipelineResult>();
                std::string serr;
                if (!runPipeline(sub, sp, reference, denoise, *sr, serr, aborted))
                {
                    if (isAborted())
                        return false;
                    continue;
                }
                mine = sr;
                Dk = &sr->beauty;
            }
            // align precisely on the clean (denoised) images
            std::vector<float> rdx(out.deflicker.dx.begin() + N * k, out.deflicker.dx.begin() + N * (k + 1));
            std::vector<float> rdy(out.deflicker.dy.begin() + N * k, out.deflicker.dy.begin() + N * (k + 1));
            {
                const PipelineFrame& nf = in.neighbours[out.dfNeighbour[k]];
                DeflickerFrame a, b;
                a.beauty = View3::planar(out.beauty.data(), out.beauty.data() + N, out.beauty.data() + 2 * N, W);
                a.albedo = in.cur.albedo;
                a.normal = in.cur.normal;
                b.beauty = View3::planar(Dk->data(), Dk->data() + N, Dk->data() + 2 * N, W);
                b.albedo = nf.albedo;
                b.normal = nf.normal;
                deflickerRefine(W, H, a, b, rdx.data(), rdy.data());
            }
            // blend where the denoised results agree
            const float tol = std::max(1e-3f, p.stabilizeTolerance);
            const float* dx = rdx.data();
            const float* dy = rdy.data();
            const float* geo = &out.deflicker.geo[N * k];
            // a real change of the light between the two frames (seen by the
            // deflicker on the noisy images) is never blended away
            const float* same = out.deflicker.same.size() >= N * (k + 1) ? &out.deflicker.same[N * k] : nullptr;
            detail::parallelFor(H, nth, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < W; ++x)
                    {
                        const size_t i = (size_t)y * W + x;
                        if (geo[i] <= 1e-3f)
                            continue;
                        const float fx = x + dx[i], fy = y + dy[i];
                        if (fx < 0.f || fy < 0.f || fx > (float)(W - 1) || fy > (float)(H - 1))
                            continue;
                        float v[3], d = 0.f;
                        bool ok = true;
                        for (int c = 0; c < 3; ++c)
                        {
                            v[c] = bilinear(Dk->data() + c * N, W, H, fx, fy);
                            ok = ok && std::isfinite(v[c]);
                            d += std::fabs(tmap(v[c]) - tmap(out.beauty[c * N + i]));
                        }
                        if (!ok)
                            continue;
                        const float q = d / (3.f * tol);
                        const float wt = geo[i] * (same ? same[i] : 1.f) * std::exp(-q * q);
                        for (int c = 0; c < 3; ++c)
                            acc[c * N + i] += wt * v[c];
                        wsum[i] += wt;
                    }
            });
        }
        detail::parallelFor(H, nth, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x)
                {
                    const size_t i = (size_t)y * W + x;
                    for (int c = 0; c < 3; ++c)
                        out.beauty[c * N + i] = acc[c * N + i] / wsum[i];
                }
        });
        out.stabilized = true;
    }
    out.seconds[2] = secondsSince(t0);

    out.valid = true;
    return true;
}

void pipelineAov(const PipelineResult& r, const PipelineParams& p, int layer, const std::array<const float*, 3>& cur,
                 const std::vector<std::array<const float*, 3>>& nb, const std::array<float*, 3>& out, const View3& albedo)
{
    const int W = r.width, H = r.height;
    const size_t N = (size_t)W * H;
    for (int c = 0; c < 3; ++c)
        if (out[c] && cur[c])
            std::memcpy(out[c], cur[c], N * sizeof(float));
    // firefly
    if (p.firefly && r.firefly.valid && layer >= 0)
    {
        OutBuffer ob;
        for (int c = 0; c < 3; ++c)
            ob.ch[c] = out[c];
        ob.width = W;
        ob.height = H;
        ob.rowStride = W;
        applyLayer(r.firefly, p.ff, layer, ob, 1.f);
        compensateLayer(r.firefly, p.ff, layer, ob, albedo, 1.f);
    }
    // deflicker, with the same offsets and weights as the beauty
    if (p.deflicker && r.deflicker.valid && r.deflicker.count > 0)
    {
        std::vector<float> tmp(N);
        for (int c = 0; c < 3; ++c)
        {
            if (!out[c])
                continue;
            std::memcpy(tmp.data(), out[c], N * sizeof(float));
            std::vector<const float*> ptr;
            std::vector<int64_t> st;
            for (int k = 0; k < r.deflicker.count; ++k)
            {
                const int idx = k < (int)r.dfNeighbour.size() ? r.dfNeighbour[k] : -1;
                ptr.push_back(idx >= 0 && idx < (int)nb.size() ? nb[idx][c] : nullptr);
                st.push_back(W);
            }
            deflickerApplyChannel(r.deflicker, p.df.amount, tmp.data(), W, ptr, st, out[c], W);
        }
    }
}

std::string pipelineAovs(const PipelineResult& r, const PipelineParams& p, const PipelineInput& in,
                         const ReferenceFn& denoise, const std::vector<int>& layers,
                         const std::vector<std::array<const float*, 3>>& cur,
                         const std::vector<std::vector<std::array<const float*, 3>>>& nb,
                         const std::vector<std::array<float*, 3>>& out)
{
    const int W = r.width, H = r.height;
    const size_t N = (size_t)W * H;
    const size_t na = std::min(cur.size(), out.size());
    const int nth = detail::threadCount(0);
    std::string warn;
    static const std::vector<std::array<const float*, 3>> none;
    // 1. firefly + deflicker; keep the stage-2 sum for the completeness test
    for (size_t j = 0; j < na; ++j)
        pipelineAov(r, p, j < layers.size() ? layers[j] : -2, cur[j], j < nb.size() ? nb[j] : none, out[j], in.cur.albedo);
    const bool beautyDenoised = !r.beautyStage2.empty();
    if (!beautyDenoised || na == 0)
        return warn;

    // 2. do the AOVs add up to the beauty (before denoising)?
    bool complete = false;
    {
        std::vector<float> rel;
        for (size_t i = 0; i < N; i += 3)
            for (int c = 0; c < 3; ++c)
            {
                const float b = r.beautyStage2[c * N + i];
                if (!std::isfinite(b) || std::fabs(b) < 1e-3f)
                    continue;
                float s = 0.f;
                for (size_t j = 0; j < na; ++j)
                    if (out[j][c])
                        s += out[j][c][i];
                rel.push_back(std::fabs(s - b) / std::fabs(b));
            }
        if (rel.size() > 100)
        {
            std::nth_element(rel.begin(), rel.begin() + rel.size() * 9 / 10, rel.end());
            complete = rel[rel.size() * 9 / 10] < 0.02f;
        }
    }

    // 3. denoise each AOV
    std::vector<std::vector<float>> A(na);
    if (p.denoiseAovs)
    {
        for (size_t j = 0; j < na; ++j)
        {
            A[j].resize(N * 3);
            for (int c = 0; c < 3; ++c)
                for (size_t i = 0; i < N; ++i)
                    A[j][c * N + i] = out[j][c] ? out[j][c][i] : 0.f;
            std::string e;
            if (!denoisePlanar(denoise, W, H, in.cur.albedo, in.cur.normal, p.denoiseAmount, A[j], e))
                warn = "AOV denoise failed (" + e + ")";
        }
    }
    else
        return warn; // AOVs keep their (firefly + deflicker) values, as before

    // 4. share what the beauty received that the AOVs did not
    const std::vector<float>& Bf = r.beauty;
    const std::vector<float>& Bd = r.stabilized ? r.beautyDenoised : r.beauty;
    detail::parallelFor(H, nth, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                const size_t i = (size_t)y * W + x;
                for (int c = 0; c < 3; ++c)
                {
                    if (complete && p.aovsMatchBeauty)
                    {
                        double sum = 0, sabs = 0;
                        for (size_t j = 0; j < na; ++j)
                        {
                            sum += A[j][c * N + i];
                            sabs += std::fabs(A[j][c * N + i]);
                        }
                        const double delta = (double)Bf[c * N + i] - sum;
                        for (size_t j = 0; j < na; ++j)
                        {
                            const double share = sabs > 1e-12 ? std::fabs(A[j][c * N + i]) / sabs : 1.0 / na;
                            if (out[j][c])
                                out[j][c][i] = (float)(A[j][c * N + i] + delta * share);
                        }
                    }
                    else
                    {
                        // pass the stabilisation on, in proportion
                        const float bd = Bd[c * N + i], delta = Bf[c * N + i] - bd;
                        for (size_t j = 0; j < na; ++j)
                        {
                            const float a = A[j][c * N + i];
                            const float share =
                                (std::fabs(bd) > 1e-6f) ? std::max(0.f, std::min(1.f, a / bd)) : 0.f;
                            if (out[j][c])
                                out[j][c][i] = a + delta * share;
                        }
                    }
                }
            }
    });
    if (!complete && p.aovsMatchBeauty && warn.empty())
        warn = "the AOVs do not add up to the beauty: each was denoised on its own";
    return warn;
}

} // namespace cleanpass
