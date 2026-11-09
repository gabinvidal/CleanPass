// CleanPass - Deflicker (motion-compensated temporal stabilisation)
// Copyright (c) 2026 Gabin Vidal. MIT License (see LICENSE).
//
// Motion estimation, per neighbour frame:
//   1. coarse block matching (large patch on a slightly blurred feature, so it
//      locks onto the real motion rather than onto the noise),
//   2. median filtering of the displacement field (removes isolated wrong
//      matches),
//   3. local refinement (+-1 px, small patch) with a parabolic sub-pixel fit.
// Each neighbour sample then gets a weight from how well its patch matches,
// relative to the noise level measured on the frame itself.

#include "cp_deflicker.h"
#include "cp_parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace cleanpass
{
namespace
{

inline float finite0(float v) { return std::isfinite(v) ? v : 0.f; }

// Matching feature: compressed luminance, so that bright and dark areas weigh
// about the same in a patch distance.
std::vector<float> feature(const View3& v, int W, int H, int threads)
{
    std::vector<float> f((size_t)W * H);
    detail::parallelFor(H, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                float l = 0.2126f * finite0(v.at(x, y, 0)) + 0.7152f * finite0(v.at(x, y, 1)) +
                          0.0722f * finite0(v.at(x, y, 2));
                l = std::max(0.f, l);
                f[(size_t)y * W + x] = l / (1.f + l);
            }
    });
    return f;
}

std::vector<float> blur3(const std::vector<float>& f, int W, int H, int threads)
{
    std::vector<float> t(f.size()), o(f.size());
    detail::parallelFor(H, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                float s = 0.f;
                int n = 0;
                for (int d = -1; d <= 1; ++d)
                    if (x + d >= 0 && x + d < W)
                    {
                        s += f[(size_t)y * W + x + d];
                        ++n;
                    }
                t[(size_t)y * W + x] = s / n;
            }
    });
    detail::parallelFor(H, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                float s = 0.f;
                int n = 0;
                for (int d = -1; d <= 1; ++d)
                    if (y + d >= 0 && y + d < H)
                    {
                        s += t[(size_t)(y + d) * W + x];
                        ++n;
                    }
                o[(size_t)y * W + x] = s / n;
            }
    });
    return o;
}

struct Guides
{
    std::vector<float> g; // 6 per pixel (albedo rgb, normal xyz * 0.5)
    bool valid = false;
};

Guides guides(const View3& alb, const View3& nrm, int W, int H, int threads)
{
    Guides G;
    if (!alb.valid() || !nrm.valid())
        return G;
    G.g.resize((size_t)W * H * 6);
    detail::parallelFor(H, threads, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                float* o = &G.g[((size_t)y * W + x) * 6];
                for (int c = 0; c < 3; ++c)
                {
                    o[c] = std::min(1.f, std::max(0.f, finite0(alb.at(x, y, c))));
                    o[3 + c] = 0.5f * std::min(1.f, std::max(-1.f, finite0(nrm.at(x, y, c))));
                }
            }
    });
    G.valid = true;
    return G;
}

const float kLambdaG = 0.5f;   // weight of the guides in the patch distance
const float kTexture = 0.5f;   // tolerance added per unit of local variance
const float kEnvBright = 0.3f; // stops a neighbour sample may exceed the local max of the current frame
const float kEnvDark = 1.0f;   // stops it may fall below the local min
const float kEnvFall = 2.0f;   // weight falloff beyond that (per stop)
const float kGuideOk = 0.02f;  // guide distance still considered the same surface
const float kGuideFall = 0.05f;

// Local mean and variance of each channel over a (2r+1)^2 box (summed-area
// tables), for the change test of the weights.
struct BoxStats
{
    std::vector<float> m[3], v[3];
};
BoxStats boxStats(const View3& img, int W, int H, int r, int nth)
{
    BoxStats b;
    const size_t N = (size_t)W * H;
    for (int c = 0; c < 3; ++c)
    {
        std::vector<double> s1((size_t)(W + 1) * (H + 1), 0.0), s2((size_t)(W + 1) * (H + 1), 0.0);
        for (int y = 0; y < H; ++y)
        {
            double r1 = 0.0, r2 = 0.0;
            for (int x = 0; x < W; ++x)
            {
                const double v = finite0(img.at(x, y, c));
                r1 += v;
                r2 += v * v;
                s1[(size_t)(y + 1) * (W + 1) + x + 1] = s1[(size_t)y * (W + 1) + x + 1] + r1;
                s2[(size_t)(y + 1) * (W + 1) + x + 1] = s2[(size_t)y * (W + 1) + x + 1] + r2;
            }
        }
        b.m[c].resize(N);
        b.v[c].resize(N);
        detail::parallelFor(H, nth, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x)
                {
                    const int xa = std::max(0, x - r), xb = std::min(W, x + r + 1);
                    const int ya = std::max(0, y - r), yb = std::min(H, y + r + 1);
                    auto S = [&](const std::vector<double>& t) {
                        return t[(size_t)yb * (W + 1) + xb] - t[(size_t)ya * (W + 1) + xb] - t[(size_t)yb * (W + 1) + xa] +
                               t[(size_t)ya * (W + 1) + xa];
                    };
                    const double n = (double)(xb - xa) * (yb - ya);
                    const double m = S(s1) / n;
                    b.m[c][(size_t)y * W + x] = (float)m;
                    b.v[c][(size_t)y * W + x] = (float)std::max(0.0, S(s2) / n - m * m);
                }
        });
    }
    return b;
}

// Per-pixel distance between cur(i) and nb(j).
inline float pixDist(const std::vector<float>& fc, const std::vector<float>& fn, const Guides& gc, const Guides& gn,
                     bool useG, float wB, size_t i, size_t j)
{
    const float a = fc[i] - fn[j];
    float D = wB * a * a;
    if (useG)
    {
        const float* g0 = &gc.g[i * 6];
        const float* g1 = &gn.g[j * 6];
        float gs = 0.f;
        for (int c = 0; c < 6; ++c)
        {
            const float e = g0[c] - g1[c];
            gs += e * e;
        }
        D += kLambdaG * gs;
    }
    return D;
}

// Coarse integer block matching with box-filtered distances. Each thread owns
// a band of rows (plus the patch halo) and runs every displacement.
void blockMatch(const std::vector<float>& fc, const std::vector<float>& fn, const Guides& gc, const Guides& gn,
                bool useG, float wB, int W, int H, int s, int pr, int nth, std::vector<int8_t>& bdx,
                std::vector<int8_t>& bdy)
{
    std::vector<std::pair<int, int>> disp;
    for (int dy = -s; dy <= s; ++dy)
        for (int dx = -s; dx <= s; ++dx)
            disp.emplace_back(dx, dy);
    std::stable_sort(disp.begin(), disp.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first * a.first + a.second * a.second < b.first * b.first + b.second * b.second;
    });
    const size_t N = (size_t)W * H;
    std::vector<float> best(N, 1e30f);
    bdx.assign(N, 0);
    bdy.assign(N, 0);

    detail::parallelFor(H, nth, [&](int y0, int y1) {
        const int ya = std::max(0, y0 - pr), yb = std::min(H, y1 + pr);
        std::vector<float> hb((size_t)(yb - ya) * W), rowD(W), col(W);
        for (const auto& d : disp)
        {
            const int dx = d.first, dy = d.second;
            for (int y = ya; y < yb; ++y)
            {
                const int yy = y + dy;
                for (int x = 0; x < W; ++x)
                {
                    const int xx = x + dx;
                    rowD[x] = (yy < 0 || yy >= H || xx < 0 || xx >= W)
                                  ? 1.f
                                  : pixDist(fc, fn, gc, gn, useG, wB, (size_t)y * W + x, (size_t)yy * W + xx);
                }
                float* h = &hb[(size_t)(y - ya) * W];
                float acc = 0.f;
                for (int x = 0; x <= std::min(W - 1, pr); ++x)
                    acc += rowD[x];
                for (int x = 0; x < W; ++x)
                {
                    h[x] = acc;
                    const int add = x + pr + 1, sub = x - pr;
                    if (add < W)
                        acc += rowD[add];
                    if (sub >= 0)
                        acc -= rowD[sub];
                }
            }
            std::fill(col.begin(), col.end(), 0.f);
            for (int y = std::max(0, y0 - pr); y <= std::min(H - 1, y0 + pr); ++y)
            {
                const float* h = &hb[(size_t)(y - ya) * W];
                for (int x = 0; x < W; ++x)
                    col[x] += h[x];
            }
            for (int y = y0; y < y1; ++y)
            {
                float* bst = &best[(size_t)y * W];
                int8_t* ox = &bdx[(size_t)y * W];
                int8_t* oy = &bdy[(size_t)y * W];
                for (int x = 0; x < W; ++x)
                    if (col[x] < bst[x])
                    {
                        bst[x] = col[x];
                        ox[x] = (int8_t)dx;
                        oy[x] = (int8_t)dy;
                    }
                const int add = y + pr + 1, sub = y - pr;
                if (add < yb)
                {
                    const float* h = &hb[(size_t)(add - ya) * W];
                    for (int x = 0; x < W; ++x)
                        col[x] += h[x];
                }
                if (sub >= ya)
                {
                    const float* h = &hb[(size_t)(sub - ya) * W];
                    for (int x = 0; x < W; ++x)
                        col[x] -= h[x];
                }
            }
        }
    });
}

// 5x5 median of an int8 field
std::vector<int8_t> median5(const std::vector<int8_t>& f, int W, int H, int nth)
{
    std::vector<int8_t> o(f.size());
    detail::parallelFor(H, nth, [&](int y0, int y1) {
        int8_t v[25];
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                int n = 0;
                for (int dy = -2; dy <= 2; ++dy)
                {
                    const int yy = y + dy;
                    if (yy < 0 || yy >= H)
                        continue;
                    for (int dx = -2; dx <= 2; ++dx)
                    {
                        const int xx = x + dx;
                        if (xx >= 0 && xx < W)
                            v[n++] = f[(size_t)yy * W + xx];
                    }
                }
                std::nth_element(v, v + n / 2, v + n);
                o[(size_t)y * W + x] = v[n / 2];
            }
    });
    return o;
}

inline float sampleBilinear(const float* img, int64_t stride, int W, int H, float fx, float fy, bool& ok)
{
    if (fx < 0.f || fy < 0.f || fx > (float)(W - 1) || fy > (float)(H - 1))
    {
        ok = false;
        return 0.f;
    }
    const int x0 = std::min(W - 2 < 0 ? 0 : W - 2, (int)fx), y0 = std::min(H - 2 < 0 ? 0 : H - 2, (int)fy);
    const int x1 = std::min(W - 1, x0 + 1), y1 = std::min(H - 1, y0 + 1);
    const float tx = fx - x0, ty = fy - y0;
    const float a = img[(int64_t)y0 * stride + x0], b = img[(int64_t)y0 * stride + x1];
    const float c = img[(int64_t)y1 * stride + x0], d = img[(int64_t)y1 * stride + x1];
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) || !std::isfinite(d))
    {
        ok = false;
        return 0.f;
    }
    ok = true;
    return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

// 2x2 box downsample of a scalar image (and of the 6-channel guides)
std::vector<float> half(const std::vector<float>& f, int W, int H, int ch, int& w2, int& h2)
{
    w2 = (W + 1) / 2;
    h2 = (H + 1) / 2;
    std::vector<float> o((size_t)w2 * h2 * ch, 0.f);
    for (int y = 0; y < h2; ++y)
        for (int x = 0; x < w2; ++x)
            for (int c = 0; c < ch; ++c)
            {
                float s = 0.f;
                int n = 0;
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx)
                    {
                        const int xx = 2 * x + dx, yy = 2 * y + dy;
                        if (xx < W && yy < H)
                        {
                            s += f[((size_t)yy * W + xx) * ch + c];
                            ++n;
                        }
                    }
                o[((size_t)y * w2 + x) * ch + c] = s / n;
            }
    return o;
}


// Patch distance between pixel (x, y) of the current frame and (x + dx, y + dy)
// of a neighbour, split into its beauty part and its guide part.
struct Matcher
{
    const std::vector<float>& fc;
    const std::vector<float>& fn;
    const Guides& gc;
    const Guides& gn;
    bool useG;
    int W, H;

    void dist(int x, int y, int dx, int dy, int rad, float& dB, float& dG) const
    {
        float accB = 0.f, accG = 0.f;
        int n = 0;
        for (int py = -rad; py <= rad; ++py)
        {
            const int y2 = y + py, yy = y2 + dy;
            if (y2 < 0 || y2 >= H)
                continue;
            for (int px = -rad; px <= rad; ++px)
            {
                const int x2 = x + px, xx = x2 + dx;
                if (x2 < 0 || x2 >= W)
                    continue;
                ++n;
                if (yy < 0 || yy >= H || xx < 0 || xx >= W)
                {
                    accB += 1.f;
                    accG += 1.f;
                    continue;
                }
                const size_t a = (size_t)y2 * W + x2, b = (size_t)yy * W + xx;
                const float e = fc[a] - fn[b];
                accB += e * e;
                if (useG)
                {
                    const float* g0 = &gc.g[a * 6];
                    const float* g1 = &gn.g[b * 6];
                    for (int q = 0; q < 6; ++q)
                    {
                        const float d = g0[q] - g1[q];
                        accG += d * d;
                    }
                }
            }
        }
        const float nn = n ? (float)n : 1.f;
        dB = accB / nn;
        dG = accG / nn;
    }

    // Best integer offset in a (2R+1)^2 window around (bx, by) with a 3x3
    // patch, plus a parabolic sub-pixel fit. Returns the cost of the best one.
    float search(int x, int y, int bx, int by, int R, float wB, float& ox, float& oy) const
    {
        int jx, jy;
        return search(x, y, bx, by, R, wB, ox, oy, jx, jy);
    }
    float search(int x, int y, int bx, int by, int R, float wB, float& ox, float& oy, int& jx, int& jy) const
    {
        float S[5][5];
        int ix = 0, iy = 0;
        float best = 1e30f;
        for (int ry = -R; ry <= R; ++ry)
            for (int rx = -R; rx <= R; ++rx)
            {
                float dB, dG;
                dist(x, y, bx + rx, by + ry, 1, dB, dG);
                const float c = wB * dB + kLambdaG * dG;
                S[ry + R][rx + R] = c;
                if (c < best - 1e-9f)
                {
                    best = c;
                    ix = rx;
                    iy = ry;
                }
            }
        float sx = 0.f, sy = 0.f;
        if (ix > -R && ix < R)
        {
            const float l = S[iy + R][ix + R - 1], c0 = S[iy + R][ix + R], r = S[iy + R][ix + R + 1];
            const float den = l - 2.f * c0 + r;
            if (den > 1e-9f)
                sx = std::max(-0.5f, std::min(0.5f, 0.5f * (l - r) / den));
        }
        if (iy > -R && iy < R)
        {
            const float u = S[iy + R - 1][ix + R], c0 = S[iy + R][ix + R], d = S[iy + R + 1][ix + R];
            const float den = u - 2.f * c0 + d;
            if (den > 1e-9f)
                sy = std::max(-0.5f, std::min(0.5f, 0.5f * (u - d) / den));
        }
        ox = (float)(bx + ix) + sx;
        oy = (float)(by + iy) + sy;
        jx = bx + ix;
        jy = by + iy;
        return best;
    }
};

// Pixels without measurable motion (conf == 0) take the motion of the
// measurable pixels around them: median per 16x16 block, empty blocks filled
// from their neighbours, then bilinear interpolation (sub-pixel and smooth).
void fillFlat(const Matcher& M, const std::vector<uint8_t>& conf, const std::vector<uint8_t>& good, int pr, int nth,
              float* dx, float* dy, float* od, float* og)
{
    const int W = M.W, H = M.H, B = 16;
    const int bw = (W + B - 1) / B, bh = (H + B - 1) / B;
    std::vector<float> mx(bw * bh, 0.f), my(bw * bh, 0.f);
    std::vector<uint8_t> have(bw * bh, 0);
    size_t nConf = 0;
    for (size_t i = 0; i < (size_t)W * H; ++i)
        nConf += conf[i];
    if (nConf == (size_t)W * H)
        return;
    detail::parallelFor(bh, nth, [&](int b0, int b1) {
        std::vector<float> vx, vy;
        for (int by = b0; by < b1; ++by)
            for (int bx = 0; bx < bw; ++bx)
            {
                vx.clear();
                vy.clear();
                for (int y = by * B; y < std::min(H, by * B + B); ++y)
                    for (int x = bx * B; x < std::min(W, bx * B + B); ++x)
                    {
                        const size_t i = (size_t)y * W + x;
                        if (good[i])
                        {
                            vx.push_back(dx[i]);
                            vy.push_back(dy[i]);
                        }
                    }
                if (vx.size() >= 8)
                {
                    std::nth_element(vx.begin(), vx.begin() + vx.size() / 2, vx.end());
                    std::nth_element(vy.begin(), vy.begin() + vy.size() / 2, vy.end());
                    mx[by * bw + bx] = vx[vx.size() / 2];
                    my[by * bw + bx] = vy[vy.size() / 2];
                    have[by * bw + bx] = 1;
                }
            }
    });
    // fill empty blocks: grow from the known ones; none known -> no motion
    bool any = false;
    for (uint8_t h : have)
        any = any || h;
    if (!any)
        std::fill(have.begin(), have.end(), 1);
    for (int it = 0; it < bw + bh; ++it)
    {
        bool changed = false;
        std::vector<uint8_t> nh = have;
        for (int by = 0; by < bh; ++by)
            for (int bx = 0; bx < bw; ++bx)
            {
                const int b = by * bw + bx;
                if (have[b])
                    continue;
                float sx = 0.f, sy = 0.f;
                int n = 0;
                for (int dy2 = -1; dy2 <= 1; ++dy2)
                    for (int dx2 = -1; dx2 <= 1; ++dx2)
                    {
                        const int xx = bx + dx2, yy = by + dy2;
                        if (xx < 0 || yy < 0 || xx >= bw || yy >= bh || !have[yy * bw + xx])
                            continue;
                        sx += mx[yy * bw + xx];
                        sy += my[yy * bw + xx];
                        ++n;
                    }
                if (n)
                {
                    mx[b] = sx / n;
                    my[b] = sy / n;
                    nh[b] = 1;
                    changed = true;
                }
            }
        have.swap(nh);
        if (!changed)
            break;
    }
    detail::parallelFor(H, nth, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                const size_t i = (size_t)y * W + x;
                if (conf[i])
                    continue;
                // bilinear between block centres
                const float gx = std::max(0.f, std::min((float)(bw - 1), (x + 0.5f) / B - 0.5f));
                const float gy = std::max(0.f, std::min((float)(bh - 1), (y + 0.5f) / B - 0.5f));
                const int x0 = (int)gx, y0b = (int)gy;
                const int x1 = std::min(bw - 1, x0 + 1), y1b = std::min(bh - 1, y0b + 1);
                const float tx = gx - x0, ty = gy - y0b;
                auto lerp2 = [&](const std::vector<float>& f) {
                    return (f[y0b * bw + x0] * (1 - tx) + f[y0b * bw + x1] * tx) * (1 - ty) +
                           (f[y1b * bw + x0] * (1 - tx) + f[y1b * bw + x1] * tx) * ty;
                };
                dx[i] = lerp2(mx);
                dy[i] = lerp2(my);
                M.dist(x, y, (int)std::floor(dx[i] + 0.5f), (int)std::floor(dy[i] + 0.5f), pr, od[i], og[i]);
            }
    });
}

inline float depthOf(const View3& v, int x, int y)
{
    const float z = std::fabs(v.at(x, y, 0));
    return (std::isfinite(z) && z < 1e9f) ? std::max(z, 1e-6f) : 1e9f;
}

// Finds how the renderer's motion vectors map to pixel displacements: the
// displacement to the frame at offset o is o * (ax * mx, ay * my). Common
// conventions are tried (pixels or NDC, full or half frame, forward or
// backward, y up or down), scored against the image itself, then the scale is
// refined by least squares. Returns false when the vectors do not explain the
// image better than no motion at all.
bool calibrateVectors(const Matcher& M, float wB, int o, const std::vector<float>& mx, const std::vector<float>& my,
                      const std::vector<float>& lvar, int nth, float& axOut, float& ayOut)
{
    const int W = M.W, H = M.H;
    const size_t N = (size_t)W * H;
    // samples: textured pixels with clearly non-zero vectors
    std::vector<float> mag;
    mag.reserve(N / 16 + 1);
    for (size_t i = 0; i < N; i += 7)
        if (std::isfinite(mx[i]) && std::isfinite(my[i]))
        {
            const float m = std::hypot(mx[i], my[i]);
            if (m > 0.f)
                mag.push_back(m);
        }
    if (mag.size() < 64)
    {
        axOut = ayOut = 1.f; // (almost) no motion: any convention gives ~0
        return true;
    }
    std::nth_element(mag.begin(), mag.begin() + mag.size() * 6 / 10, mag.end());
    const float mThr = mag[mag.size() * 6 / 10];
    std::vector<float> lv;
    for (size_t i = 0; i < N; i += 7)
        if (lvar[i] > 0.f)
            lv.push_back(lvar[i]);
    float lvThr = 1e-12f;
    if (lv.size() > 64)
    {
        std::nth_element(lv.begin(), lv.begin() + lv.size() / 2, lv.end());
        lvThr = std::max(lvThr, lv[lv.size() / 2]);
    }
    const int margin = 4;
    std::vector<int> samp;
    for (size_t i = 0; i < N; i += 3)
    {
        const int x = (int)(i % W), y = (int)(i / W);
        if (x < margin || y < margin || x >= W - margin || y >= H - margin)
            continue;
        if (!std::isfinite(mx[i]) || !std::isfinite(my[i]) || std::hypot(mx[i], my[i]) < mThr || lvar[i] < lvThr)
            continue;
        samp.push_back((int)i);
    }
    if (samp.size() > 6000)
    {
        std::vector<int> t;
        const size_t st = samp.size() / 6000 + 1;
        for (size_t k = 0; k < samp.size(); k += st)
            t.push_back(samp[k]);
        samp.swap(t);
    }
    if (samp.size() < 50)
    {
        axOut = ayOut = 1.f;
        return true;
    }
    // candidates
    std::vector<std::pair<float, float>> cand;
    const float bases[][2] = {{1.f, 1.f}, {0.5f, 0.5f}, {2.f, 2.f}, {0.5f * W, 0.5f * H}, {(float)W, (float)H}};
    for (const auto& bs : bases)
        for (int sxs = -1; sxs <= 1; sxs += 2)
            for (int sys = -1; sys <= 1; sys += 2)
                cand.emplace_back(sxs * bs[0], sys * bs[1]);
    cand.emplace_back(0.f, 0.f); // no motion (reference)
    const size_t nc = cand.size();
    std::vector<float> score(nc);
    for (size_t c = 0; c < nc; ++c)
    {
        std::vector<float> d(samp.size());
        detail::parallelFor((int)samp.size(), nth, [&](int b0, int b1) {
            for (int k = b0; k < b1; ++k)
            {
                const int i = samp[k], x = i % W, y = i / W;
                const float px = o * cand[c].first * mx[i], py = o * cand[c].second * my[i];
                if (!std::isfinite(px) || !std::isfinite(py) || std::fabs(px) > W || std::fabs(py) > H)
                {
                    d[k] = 1e3f;
                    continue;
                }
                float dB, dG;
                M.dist(x, y, (int)std::lround(px), (int)std::lround(py), 1, dB, dG);
                d[k] = wB * dB + kLambdaG * dG;
            }
        });
        std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
        score[c] = d[d.size() / 2];
    }
    size_t best = 0;
    for (size_t c = 1; c + 1 < nc; ++c)
        if (score[c] < score[best])
            best = c;
    if (!(score[best] < 0.8f * score[nc - 1]))
        return false;
    // least-squares refinement of the scale on each axis (on large motions only:
    // on small ones the sub-pixel fit is biased towards whole pixels)
    float ax = cand[best].first, ay = cand[best].second;
    for (int it = 0; it < 2; ++it)
    {
        double sxp = 0, sxx = 0, syp = 0, syy = 0;
        int nx = 0, ny = 0;
        std::vector<float> ex(samp.size()), ey(samp.size()), cost(samp.size());
        detail::parallelFor((int)samp.size(), nth, [&](int b0, int b1) {
            for (int k = b0; k < b1; ++k)
            {
                const int i = samp[k], x = i % W, y = i / W;
                const float px = o * ax * mx[i], py = o * ay * my[i];
                float ox = 0.f, oy = 0.f;
                cost[k] = (std::fabs(px) > W || std::fabs(py) > H)
                              ? 1e3f
                              : M.search(x, y, (int)std::lround(px), (int)std::lround(py), 2, wB, ox, oy);
                ex[k] = ox;
                ey[k] = oy;
            }
        });
        std::vector<float> cs = cost;
        std::nth_element(cs.begin(), cs.begin() + cs.size() / 2, cs.end());
        const float cThr = 1.5f * cs[cs.size() / 2];
        for (size_t k = 0; k < samp.size(); ++k)
        {
            if (cost[k] > cThr)
                continue;
            const int i = samp[k];
            const double rx = (double)o * mx[i], ry = (double)o * my[i];
            if (std::fabs(rx * ax) >= 4.0)
            {
                sxp += ex[k] * rx;
                sxx += rx * rx;
                ++nx;
            }
            if (std::fabs(ry * ay) >= 4.0)
            {
                syp += ey[k] * ry;
                syy += ry * ry;
                ++ny;
            }
        }
        if (nx > 100 && sxx > 0)
        {
            const float a = (float)(sxp / sxx);
            if (a * ax > 0.f)
                ax = std::max(0.5f * std::fabs(ax), std::min(2.f * std::fabs(ax), std::fabs(a))) * (ax > 0 ? 1.f : -1.f);
        }
        if (ny > 100 && syy > 0)
        {
            const float a = (float)(syp / syy);
            if (a * ay > 0.f)
                ay = std::max(0.5f * std::fabs(ay), std::min(2.f * std::fabs(ay), std::fabs(a))) * (ay > 0 ? 1.f : -1.f);
        }
    }
    // Renderers write vectors in whole conventions (pixels or NDC, over the
    // full shutter or a fraction of it: x2 for Arnold's default half-frame
    // shutter). The fit above is a little noisy (+-10 % from frame to frame) and
    // biased towards whole pixels, so a scale close to such a value is snapped
    // to it: the same, exact scale on every frame, whatever the render order.
    const bool ndc = std::fabs(cand[best].first) > 3.f;
    auto snap = [&](float a, float unit) {
        const float nice[] = {0.25f, 0.5f, 1.f, 2.f, 4.f};
        for (float n : nice)
            if (std::fabs(std::fabs(a) / (n * unit) - 1.f) < 0.15f)
                return n * unit * (a > 0.f ? 1.f : -1.f);
        return a;
    };
    axOut = snap(ax, ndc ? 0.5f * W : 1.f);
    ayOut = snap(ay, ndc ? 0.5f * H : 1.f);
    return true;
}

} // namespace

bool deflickerAnalyze(int W, int H, const DeflickerFrame& cur, const std::vector<DeflickerFrame>& nbs,
                      const DeflickerParams& pIn, DeflickerResult& out, const AbortFn& aborted)
{
    const auto t0 = std::chrono::steady_clock::now();
    out = DeflickerResult();
    out.width = W;
    out.height = H;
    if (W <= 0 || H <= 0 || !cur.beauty.valid())
        return true;

    DeflickerParams p = pIn;
    p.search = std::max(0, std::min(8, p.search));
    p.patch = std::max(1, std::min(4, p.patch));
    p.tolerance = std::max(0.01f, p.tolerance);
    const int nth = detail::threadCount(p.threads);
    const size_t N = (size_t)W * H;
    auto isAborted = [&]() { return aborted && aborted(); };

    std::vector<const DeflickerFrame*> used;
    for (const auto& n : nbs)
        if (n.beauty.valid())
            used.push_back(&n);
    out.count = (int)used.size();
    out.dx.assign(N * out.count, 0.f);
    out.dy.assign(N * out.count, 0.f);
    out.weight.assign(N * out.count, 0.f);
    out.geo.assign(N * out.count, 0.f);
    out.same.assign(N * out.count, 0.f);
    if (out.count == 0)
    {
        out.valid = true;
        return true;
    }
    std::vector<float> dist(N * out.count, 0.f), gdist(N * out.count, 0.f);

    const std::vector<float> fc = feature(cur.beauty, W, H, nth);
    const std::vector<float> fcb = blur3(fc, W, H, nth);
    const Guides gc = p.useGuides ? guides(cur.albedo, cur.normal, W, H, nth) : Guides();
    const int pr = p.patch;
    // local contrast of the current frame: on textured areas a correct match
    // still leaves some residual (sub-pixel motion), so the tolerance grows there
    std::vector<float> lvar(N, 0.f);
    detail::parallelFor(H, nth, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                double s = 0, s2 = 0;
                int n = 0;
                for (int py = std::max(0, y - pr); py <= std::min(H - 1, y + pr); ++py)
                    for (int px = std::max(0, x - pr); px <= std::min(W - 1, x + pr); ++px)
                    {
                        const float v = fc[(size_t)py * W + px];
                        s += v;
                        s2 += (double)v * v;
                        ++n;
                    }
                const double m = s / n;
                lvar[(size_t)y * W + x] = (float)std::max(0.0, s2 / n - m * m);
            }
    });
    // Brightness envelope of the current frame (3x3 min / max energy). A
    // neighbour sample far above it is a firefly of that frame - the current
    // one has been cleaned, its neighbours have not - and must not be blended
    // back in (the same idea as the history clamp of temporal anti-aliasing,
    // applied as a weight so that the AOVs stay linear).
    std::vector<float> eCur(N), eLo(N), eHi(N);
    detail::parallelFor(H, nth, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                float e = 0.f;
                for (int c = 0; c < 3; ++c)
                    e += std::max(0.f, finite0(cur.beauty.at(x, y, c)));
                eCur[(size_t)y * W + x] = e;
            }
    });
    detail::parallelFor(H, nth, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                float lo = std::numeric_limits<float>::max(), hi = 0.f;
                for (int py = std::max(0, y - 1); py <= std::min(H - 1, y + 1); ++py)
                    for (int px = std::max(0, x - 1); px <= std::min(W - 1, x + 1); ++px)
                    {
                        const float e = eCur[(size_t)py * W + px];
                        lo = std::min(lo, e);
                        hi = std::max(hi, e);
                    }
                eLo[(size_t)y * W + x] = lo;
                eHi[(size_t)y * W + x] = hi;
            }
    });
    const int prCoarse = std::min(6, pr + 2);

    // ---- fixed render seed ------------------------------------------------
    // With a fixed seed, static areas are bit-identical from one frame to the
    // next. Averaging them changes nothing (harmless), but they must not drive
    // the noise estimate, and the matching must lean on the guides rather than
    // on a noise pattern that stays put on screen.
    {
        double ident = 0, sig = 0;
        for (int k = 0; k < out.count; ++k)
        {
            const DeflickerFrame& nb = *used[k];
            for (size_t i = 0; i < N; i += 3)
            {
                const int x = (int)(i % W), y = (int)(i / W);
                if (eCur[i] < 1e-4f)
                    continue;
                sig += 1;
                bool same = true;
                for (int c = 0; c < 3 && same; ++c)
                {
                    const float a = cur.beauty.at(x, y, c), b = nb.beauty.at(x, y, c);
                    same = std::fabs(a - b) <= 1e-6f * (1.f + std::fabs(a));
                }
                ident += same ? 1 : 0;
            }
        }
        out.identical = sig > 0 ? (float)(ident / sig) : 0.f;
        out.fixedSeed = out.identical > 0.05f;
    }
    float wB = (out.fixedSeed && gc.valid) ? 0.1f : 1.f;
    // guide flatness of the current frame (for the screen-locked noise test)
    std::vector<uint8_t> flat;
    if (gc.valid)
    {
        flat.assign(N, 0);
        detail::parallelFor(H, nth, [&](int y0, int y1) {
            for (int y = std::max(1, y0); y < std::min(H - 1, y1); ++y)
                for (int x = 1; x < W - 1; ++x)
                {
                    const float* g0 = &gc.g[((size_t)y * W + x) * 6];
                    float d = 0.f;
                    for (int py = -1; py <= 1; ++py)
                        for (int px = -1; px <= 1; ++px)
                        {
                            const float* g1 = &gc.g[((size_t)(y + py) * W + x + px) * 6];
                            for (int q = 0; q < 6; ++q)
                                d = std::max(d, std::fabs(g1[q] - g0[q]));
                        }
                    flat[(size_t)y * W + x] = d < 0.01f && fc[(size_t)y * W + x] > 0.01f;
                }
        });
    }

    // ---- optional passes of the current frame -------------------------------
    const bool haveMV = p.useVectors && cur.motion.valid();
    std::vector<float> mvx, mvy;
    if (haveMV)
    {
        mvx.resize(N);
        mvy.resize(N);
        detail::parallelFor(H, nth, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x)
                {
                    mvx[(size_t)y * W + x] = cur.motion.at(x, y, 0);
                    mvy[(size_t)y * W + x] = cur.motion.at(x, y, 1);
                }
        });
    }
    bool vecOk = false, vecTried = false;
    float vax = 0.f, vay = 0.f;
    size_t vecPicked = 0, vecTotal = 0;

    // Where the motion can be measured: texture clearly above the noise in
    // the blurred feature, or detail in the noise-free guides. Elsewhere (flat
    // areas) block matching only fits the noise; those pixels take the motion
    // of the measurable pixels around them instead.
    std::vector<uint8_t> conf(N, 0);
    float sn2 = 0.f; // pixel noise variance of the matching feature
    {
        std::vector<float> hp;
        for (size_t i = 0; i < N; i += 3)
            if (fc[i] > 0.01f)
                hp.push_back(std::fabs(fc[i] - fcb[i]));
        if (!hp.empty())
        {
            std::nth_element(hp.begin(), hp.begin() + hp.size() / 2, hp.end());
            const float s = 1.4826f * hp[hp.size() / 2] * 1.5f; // MAD of the high-pass ~ 2/3 of the pixel noise
            sn2 = s * s;
        }
        const float thrB = 3.f * sn2 / 9.f; // variance of a 3x3-blurred noise ~ sigma^2 / 9
        const int rr = 3;
        detail::parallelFor(H, nth, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x)
                {
                    double s1 = 0, s2 = 0;
                    int n = 0;
                    float gmin[6], gmax[6];
                    for (int q = 0; q < 6; ++q)
                    {
                        gmin[q] = 1e9f;
                        gmax[q] = -1e9f;
                    }
                    for (int py = std::max(0, y - rr); py <= std::min(H - 1, y + rr); ++py)
                        for (int px = std::max(0, x - rr); px <= std::min(W - 1, x + rr); ++px)
                        {
                            const size_t j = (size_t)py * W + px;
                            s1 += fcb[j];
                            s2 += (double)fcb[j] * fcb[j];
                            ++n;
                            if (gc.valid)
                                for (int q = 0; q < 6; ++q)
                                {
                                    gmin[q] = std::min(gmin[q], gc.g[j * 6 + q]);
                                    gmax[q] = std::max(gmax[q], gc.g[j * 6 + q]);
                                }
                        }
                    const double m = s1 / n;
                    bool c = (s2 / n - m * m) > thrB;
                    if (!c && gc.valid)
                        for (int q = 0; q < 6 && !c; ++q)
                            c = gmax[q] - gmin[q] > 0.03f;
                    conf[(size_t)y * W + x] = c;
                }
        });
    }

    // From here on, "texture" means the local variance of the signal: the
    // pixel noise is taken out, otherwise a noisy area would look textured and
    // loosen the tolerance for wrong matches.
    for (size_t i = 0; i < N; ++i)
        lvar[i] = std::max(0.f, lvar[i] - sn2);

    // process a +-1 neighbour first: it calibrates the motion vectors
    std::vector<int> order(out.count);
    for (int k = 0; k < out.count; ++k)
        order[k] = k;
    // (a fixed order - t-1 before t+1 - so that the same frames always give the
    // same result, whatever order the caller lists them in)
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const int oa = used[a]->offset, ob = used[b]->offset;
        return std::abs(oa) != std::abs(ob) ? std::abs(oa) < std::abs(ob) : oa < ob;
    });

    for (int k : order)
    {
        if (isAborted())
            return false;
        const DeflickerFrame& nb = *used[k];
        const int o = nb.offset == 0 ? 1 : nb.offset;
        const std::vector<float> fn = feature(nb.beauty, W, H, nth);
        const std::vector<float> fnb = blur3(fn, W, H, nth);
        const Guides gn = gc.valid ? guides(nb.albedo, nb.normal, W, H, nth) : Guides();
        const bool useG = gc.valid && gn.valid;
        const int s = std::min(8, p.search * std::max(1, std::abs(nb.offset)));
        const Matcher M{fc, fn, gc, gn, useG, W, H};

        // Screen-locked noise (fixed render seed with a moving camera): on flat
        // surfaces the high frequencies are pure noise, uncorrelated between two
        // frames with an animated seed, strongly correlated with a fixed one.
        if (k == order[0] && !flat.empty())
        {
            double sxy = 0, sxx = 0, syy = 0;
            size_t n = 0;
            for (size_t i = 0; i < N; ++i)
                if (flat[i])
                {
                    const double a = fc[i] - fcb[i], b = fn[i] - fnb[i];
                    sxy += a * b;
                    sxx += a * a;
                    syy += b * b;
                    ++n;
                }
            if (n > 500 && sxx > 0 && syy > 0)
            {
                out.noiseCorrelation = (float)(sxy / std::sqrt(sxx * syy));
                if (out.noiseCorrelation > 0.3f)
                    out.fixedSeed = true;
            }
            if (out.fixedSeed && useG)
                wB = 0.1f;
        }

        // 1. coarse matching (at half resolution on large frames) + 2. median-filtered field
        std::vector<int8_t> cdx, cdy;
        if (W >= 64 && H >= 64)
        {
            int w2, h2;
            const std::vector<float> c2 = half(fcb, W, H, 1, w2, h2), n2 = half(fnb, W, H, 1, w2, h2);
            Guides gc2, gn2;
            if (useG)
            {
                gc2.g = half(gc.g, W, H, 6, w2, h2);
                gn2.g = half(gn.g, W, H, 6, w2, h2);
                gc2.valid = gn2.valid = true;
            }
            std::vector<int8_t> hdx, hdy;
            int s2 = (s + 1) / 2;
            blockMatch(c2, n2, gc2, gn2, useG, wB, w2, h2, s2, std::max(1, prCoarse / 2), nth, hdx, hdy);
            // Without vectors, many measurable pixels stuck at the edge of the
            // search window mean the motion is larger than the search: widen it.
            if (!haveMV)
                for (int pass = 0; pass < 2 && s2 < 8; ++pass)
                {
                    size_t edge = 0, tot = 0;
                    for (int y = 0; y < h2; ++y)
                        for (int x = 0; x < w2; ++x)
                        {
                            const size_t fi = (size_t)std::min(H - 1, 2 * y) * W + std::min(W - 1, 2 * x);
                            if (!conf[fi])
                                continue;
                            ++tot;
                            const size_t j = (size_t)y * w2 + x;
                            edge += (std::abs(hdx[j]) == s2 || std::abs(hdy[j]) == s2);
                        }
                    if (tot < 64 || edge * 5 < tot)
                        break;
                    s2 = std::min(8, s2 * 2);
                    blockMatch(c2, n2, gc2, gn2, useG, wB, w2, h2, s2, std::max(1, prCoarse / 2), nth, hdx, hdy);
                }
            cdx.resize(N);
            cdy.resize(N);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                {
                    const size_t j = (size_t)(y / 2) * w2 + x / 2;
                    cdx[(size_t)y * W + x] = (int8_t)std::max(-127, std::min(127, 2 * hdx[j]));
                    cdy[(size_t)y * W + x] = (int8_t)std::max(-127, std::min(127, 2 * hdy[j]));
                }
        }
        else
            blockMatch(fcb, fnb, gc, gn, useG, wB, W, H, s, prCoarse, nth, cdx, cdy);
        if (isAborted())
            return false;
        cdx = median5(cdx, W, H, nth);
        cdy = median5(cdy, W, H, nth);

        // 3. refinement +-1 px with a 3x3 patch, then sub-pixel parabola
        float* odx = &out.dx[N * k];
        float* ody = &out.dy[N * k];
        float* od = &dist[N * k];
        float* og = &gdist[N * k];
        detail::parallelFor(H, nth, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x)
                {
                    const size_t i = (size_t)y * W + x;
                    float ox, oy;
                    int jx, jy;
                    M.search(x, y, cdx[i], cdy[i], 1, wB, ox, oy, jx, jy);
                    odx[i] = ox;
                    ody[i] = oy;
                    M.dist(x, y, jx, jy, pr, od[i], og[i]);
                }
        });
        {
            // the motion of the measurable pixels whose match is also good
            // (within the noise): a match beyond the search range is not
            std::vector<uint8_t> good(N);
            const float thrB = 3.f * sn2;
            for (size_t i = 0; i < N; ++i)
                good[i] = conf[i] && od[i] <= thrB + kTexture * lvar[i] && (!useG || og[i] <= 4.f * kGuideOk);
            fillFlat(M, conf, good, pr, nth, odx, ody, od, og);
        }

        // 4. motion vectors: a second candidate per pixel, kept when it matches
        //    at least about as well (so they win on flat areas and fast motion,
        //    and lose where they describe the surface but not what it shows:
        //    reflections, refraction, transparency)
        if (haveMV)
        {
            if (!vecTried)
            {
                vecTried = true;
                vecOk = calibrateVectors(M, wB, o, mvx, mvy, lvar, nth, vax, vay);
            }
            if (vecOk)
            {
                const int R = std::abs(o) >= 2 ? 2 : 1;
                // How much worse than the best match the vectors may be and still
                // be trusted: the spread of a patch distance made of noise alone
                // (2 sn2 per pixel, averaged over the 9 or 25 pixels of the patch).
                const float margin3 = wB * 2.f * sn2 * 1.5f + 1e-7f;
                const float margin5 = wB * 2.f * sn2 * 0.8f + 1e-7f;
                std::vector<uint8_t> picked(N, 0);
                detail::parallelFor(H, nth, [&](int y0, int y1) {
                    for (int y = y0; y < y1; ++y)
                        for (int x = 0; x < W; ++x)
                        {
                            const size_t i = (size_t)y * W + x;
                            const float px = o * vax * mvx[i], py = o * vay * mvy[i];
                            if (!std::isfinite(px) || !std::isfinite(py) || std::fabs(px) > W || std::fabs(py) > H)
                                continue;
                            float ox, oy;
                            int jx, jy;
                            const int rx = (int)std::lround(px), ry = (int)std::lround(py);
                            const float best = M.search(x, y, rx, ry, R, wB, ox, oy, jx, jy);
                            // the image agrees with the vectors: keep their exact (sub-pixel) value
                            float cB, cG;
                            M.dist(x, y, rx, ry, 1, cB, cG);
                            if (wB * cB + kLambdaG * cG <= best + margin3)
                            {
                                ox = px;
                                oy = py;
                                jx = rx;
                                jy = ry;
                            }
                            float dB, dG;
                            M.dist(x, y, jx, jy, pr, dB, dG);
                            const float costV = wB * dB + kLambdaG * dG;
                            const float costA = wB * od[i] + kLambdaG * og[i];
                            if (costV <= costA + margin5)
                            {
                                odx[i] = ox;
                                ody[i] = oy;
                                od[i] = dB;
                                og[i] = dG;
                                picked[i] = 1;
                            }
                        }
                });
                for (size_t i = 0; i < N; ++i)
                    vecPicked += picked[i];
                vecTotal += N;
            }
        }
    }
    if (isAborted())
        return false;
    out.vectorsUsed = haveMV && vecOk;
    out.vecX = vax;
    out.vecY = vay;
    out.vectorShare = vecTotal ? (float)vecPicked / (float)vecTotal : 0.f;

    // Noise level: the typical residual of a good match on pixels with signal
    // (identical pixels of a fixed seed excluded).
    std::vector<float> sample;
    const size_t total = N * out.count;
    const size_t stride = std::max<size_t>(1, total / 400000);
    for (size_t i = 0; i < total; i += stride)
        if (fc[i % N] > 0.01f && dist[i] > 1e-9f)
            sample.push_back(dist[i]);
    float sigma2 = 1e-5f;
    if (!sample.empty())
    {
        std::nth_element(sample.begin(), sample.begin() + sample.size() / 2, sample.end());
        sigma2 = std::max(1e-6f, sample[sample.size() / 2]);
    }
    out.noise = std::sqrt(sigma2);

    // Per-pixel noise level from a variance pass: the variance is carried into
    // the matching feature (compressed luminance), averaged over the patch, and
    // its unknown normalisation (variance of the samples or of the pixel mean)
    // is calibrated against the matching residual on flat areas.
    std::vector<float> sig;
    if (p.useVariance && cur.variance.valid())
    {
        std::vector<float> vf(N, 0.f);
        detail::parallelFor(H, nth, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x)
                {
                    float l = 0.2126f * finite0(cur.beauty.at(x, y, 0)) + 0.7152f * finite0(cur.beauty.at(x, y, 1)) +
                              0.0722f * finite0(cur.beauty.at(x, y, 2));
                    l = std::max(0.f, l);
                    const float v = 0.2126f * 0.2126f * std::max(0.f, finite0(cur.variance.at(x, y, 0))) +
                                    0.7152f * 0.7152f * std::max(0.f, finite0(cur.variance.at(x, y, 1))) +
                                    0.0722f * 0.0722f * std::max(0.f, finite0(cur.variance.at(x, y, 2)));
                    const float d = 1.f / ((1.f + l) * (1.f + l));
                    vf[(size_t)y * W + x] = v * d * d;
                }
        });
        std::vector<float> vfp(N, 0.f);
        detail::parallelFor(H, nth, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x)
                {
                    double s = 0;
                    int n = 0;
                    for (int py = std::max(0, y - pr); py <= std::min(H - 1, y + pr); ++py)
                        for (int px = std::max(0, x - pr); px <= std::min(W - 1, x + pr); ++px)
                        {
                            s += vf[(size_t)py * W + px];
                            ++n;
                        }
                    vfp[(size_t)y * W + x] = (float)(s / n);
                }
        });
        std::vector<float> lvs;
        for (size_t i = 0; i < N; i += 5)
            lvs.push_back(lvar[i]);
        std::nth_element(lvs.begin(), lvs.begin() + lvs.size() / 2, lvs.end());
        const float flat = lvs[lvs.size() / 2];
        std::vector<float> ratio;
        for (size_t i = 0; i < total; i += stride)
        {
            const size_t q = i % N;
            if (fc[q] > 0.01f && dist[i] > 1e-9f && lvar[q] <= flat && vfp[q] > 0.f && gdist[i] < kGuideOk)
                ratio.push_back(dist[i] / (2.f * vfp[q]));
        }
        if (ratio.size() > 200)
        {
            std::nth_element(ratio.begin(), ratio.begin() + ratio.size() / 2, ratio.end());
            const float kv = ratio[ratio.size() / 2];
            if (std::isfinite(kv) && kv > 0.f)
            {
                sig.resize(N);
                for (size_t i = 0; i < N; ++i)
                    sig[i] = std::max(0.05f * sigma2, std::min(20.f * sigma2, kv * 2.f * vfp[i]));
                out.varianceUsed = true;
            }
        }
    }

    // depth envelope of the current frame (3x3), for the disocclusion test
    const bool haveZ = p.useDepth && cur.depth.valid();
    std::vector<float> zLo, zHi;
    if (haveZ)
    {
        zLo.resize(N);
        zHi.resize(N);
        detail::parallelFor(H, nth, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x)
                {
                    float lo = 1e30f, hi = 0.f;
                    for (int py = std::max(0, y - 1); py <= std::min(H - 1, y + 1); ++py)
                        for (int px = std::max(0, x - 1); px <= std::min(W - 1, x + 1); ++px)
                        {
                            const float z = depthOf(cur.depth, px, py);
                            lo = std::min(lo, z);
                            hi = std::max(hi, z);
                        }
                    zLo[(size_t)y * W + x] = std::log(lo);
                    zHi[(size_t)y * W + x] = std::log(hi);
                }
        });
    }

    // Weights
    const float invTol2 = 1.f / (p.tolerance * p.tolerance);
    // Change test: the mean of a 5x5 window, channel by channel, must agree
    // with the neighbour's within its standard error. A light switching on or
    // off, a moving shadow or a colour change shows up there clearly even
    // when the per-pixel noise is larger than the change.
    const int kBox = 2;
    const float boxN = (float)((2 * kBox + 1) * (2 * kBox + 1));
    const BoxStats bc = boxStats(cur.raw.valid() ? cur.raw : cur.beauty, W, H, kBox, nth);
    for (int k = 0; k < out.count; ++k)
    {
        const DeflickerFrame& nb = *used[k];
        const BoxStats bn = boxStats(nb.beauty, W, H, kBox, nth);
        float* sm = &out.same[N * k];
        const float* od = &dist[N * k];
        const float* og = &gdist[N * k];
        const bool useGuidesAny = gc.valid;
        const bool zTest = haveZ && nb.depth.valid();
        if (zTest)
            out.depthUsed = true;
        const float zAllow = 0.01f + 0.03f * std::abs(nb.offset);
        const float* odx = &out.dx[N * k];
        const float* ody = &out.dy[N * k];
        float* w = &out.weight[N * k];
        float* gw = &out.geo[N * k];
        detail::parallelFor(H, nth, [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y)
                for (int x = 0; x < W; ++x)
                {
                    const size_t i = (size_t)y * W + x;
                    const float fx = x + odx[i], fy = y + ody[i];
                    if (fx < 0.f || fy < 0.f || fx > (float)(W - 1) || fy > (float)(H - 1))
                    {
                        w[i] = gw[i] = sm[i] = 0.f;
                        continue;
                    }
                    // patch similarity relative to the noise level (+ local texture)
                    const float s2 = sig.empty() ? sigma2 : sig[i];
                    const float r = std::max(0.f, od[i] / (s2 + kTexture * lvar[i]) - 1.f);
                    float wt = std::exp(-0.5f * r * invTol2);
                    float geo = 1.f;
                    // the guides are noise free: a clear mismatch means another surface
                    if (useGuidesAny)
                    {
                        const float g = std::max(0.f, og[i] - kGuideOk);
                        geo *= std::exp(-g / kGuideFall);
                    }
                    // the matched sample itself must not be an outlier (firefly in
                    // the neighbour frame, or a real change of lighting)
                    const int xx = std::min(W - 1, (int)std::lround(fx)), yy = std::min(H - 1, (int)std::lround(fy));
                    float ec = 0.f, en = 0.f;
                    bool bad = false;
                    for (int c = 0; c < 3; ++c)
                    {
                        const float vn = nb.beauty.at(xx, yy, c);
                        bad = bad || !std::isfinite(vn);
                        en += std::max(0.f, finite0(vn));
                        ec += std::max(0.f, finite0(cur.beauty.at(x, y, c)));
                    }
                    const float lr = std::fabs(std::log2((en + 1e-3f) / (ec + 1e-3f)));
                    const float e = std::max(0.f, lr - 1.5f) * 1.5f;
                    wt *= std::exp(-e * e);
                    // ... nor outside the brightness envelope of the current
                    // frame (every pixel of the bilinear footprint is checked)
                    {
                        const int x0 = std::max(0, std::min(W - 1, (int)std::floor(fx)));
                        const int y0 = std::max(0, std::min(H - 1, (int)std::floor(fy)));
                        const int x1 = std::min(W - 1, x0 + 1), y1 = std::min(H - 1, y0 + 1);
                        float hiN = 0.f;
                        for (int q = 0; q < 4; ++q)
                        {
                            const int qx = (q & 1) ? x1 : x0, qy = (q & 2) ? y1 : y0;
                            float eq = 0.f;
                            for (int c = 0; c < 3; ++c)
                                eq += std::max(0.f, finite0(nb.beauty.at(qx, qy, c)));
                            hiN = std::max(hiN, eq);
                        }
                        float env = 0.f;
                        if (hiN > eHi[i])
                            env = std::max(0.f, std::log2((hiN + 1e-3f) / (eHi[i] + 1e-3f)) - kEnvBright);
                        else if (en < eLo[i])
                            env = std::max(0.f, std::log2((eLo[i] + 1e-3f) / (en + 1e-3f)) - kEnvDark);
                        env *= kEnvFall;
                        wt *= std::exp(-env * env);
                    }
                    // ... nor come from another depth (disocclusion)
                    if (zTest)
                    {
                        const float zn = std::log(depthOf(nb.depth, xx, yy));
                        const float dz = std::max(0.f, std::max(zLo[i] - zAllow - zn, zn - zHi[i] - zAllow));
                        const float q = dz / 0.03f;
                        geo *= std::exp(-q * q);
                    }
                    // ... nor be a real change of the light (5x5 means, every channel)
                    float same = 1.f;
                    {
                        const size_t j = (size_t)yy * W + xx;
                        float z = 0.f;
                        for (int c = 0; c < 3; ++c)
                        {
                            const float mc = bc.m[c][i], mn = bn.m[c][j];
                            const float floor = 1e-4f + 0.01f * 0.5f * (std::fabs(mc) + std::fabs(mn));
                            const float se = std::sqrt((bc.v[c][i] + bn.v[c][j]) / boxN + floor * floor);
                            z = std::max(z, std::fabs(mn - mc) / se);
                        }
                        const float q = std::max(0.f, z - 4.f) / 2.f;
                        same = std::exp(-q * q);
                    }
                    sm[i] = bad ? 0.f : same;
                    w[i] = bad ? 0.f : wt * geo;
                    gw[i] = bad ? 0.f : geo;
                }
        });
        // A change is a region, not a pixel: where the light changed, the few
        // pixels whose test passed by chance must not let it through either.
        {
            std::vector<float> t(N);
            const int r = 2;
            detail::parallelFor(H, nth, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < W; ++x)
                    {
                        float m = 1.f;
                        for (int q = std::max(0, x - r); q <= std::min(W - 1, x + r); ++q)
                            m = std::min(m, sm[(size_t)y * W + q]);
                        t[(size_t)y * W + x] = m;
                    }
            });
            detail::parallelFor(H, nth, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y)
                    for (int x = 0; x < W; ++x)
                    {
                        float m = 1.f;
                        for (int q = std::max(0, y - r); q <= std::min(H - 1, y + r); ++q)
                            m = std::min(m, t[(size_t)q * W + x]);
                        const size_t i = (size_t)y * W + x;
                        sm[i] = m;
                        w[i] *= m;
                    }
            });
        }
    }

    out.valid = true;
    out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

void deflickerRefine(int W, int H, const DeflickerFrame& cur, const DeflickerFrame& nb, float* dx, float* dy, int threads)
{
    if (W <= 0 || H <= 0 || !cur.beauty.valid() || !nb.beauty.valid())
        return;
    const int nth = detail::threadCount(threads);
    const std::vector<float> fc = feature(cur.beauty, W, H, nth), fn = feature(nb.beauty, W, H, nth);
    const Guides gc = guides(cur.albedo, cur.normal, W, H, nth);
    const Guides gn = gc.valid ? guides(nb.albedo, nb.normal, W, H, nth) : Guides();
    const Matcher M{fc, fn, gc, gn, gc.valid && gn.valid, W, H};
    detail::parallelFor(H, nth, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                const size_t i = (size_t)y * W + x;
                if (!std::isfinite(dx[i]) || !std::isfinite(dy[i]))
                    continue;
                float ox, oy;
                int jx, jy;
                const int rx = (int)std::lround(dx[i]), ry = (int)std::lround(dy[i]);
                const float best = M.search(x, y, rx, ry, 1, 1.f, ox, oy, jx, jy);
                if (jx == rx && jy == ry)
                    continue; // confirmed: keep the given sub-pixel value
                float cB, cG;
                M.dist(x, y, rx, ry, 1, cB, cG);
                if (cB + kLambdaG * cG > 1.5f * best + 1e-6f)
                {
                    dx[i] = ox;
                    dy[i] = oy;
                }
            }
    });
}

void deflickerApplyChannel(const DeflickerResult& r, float amount, const float* cur, int64_t curStride,
                           const std::vector<const float*>& nb, const std::vector<int64_t>& nbStride, float* out,
                           int64_t outStride)
{
    const int W = r.width, H = r.height;
    const size_t N = (size_t)W * H;
    amount = std::max(0.f, std::min(1.f, amount));
    detail::parallelFor(H, detail::threadCount(0), [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < W; ++x)
            {
                const size_t i = (size_t)y * W + x;
                const float c = cur[(int64_t)y * curStride + x];
                float o = c;
                if (r.valid && amount > 0.f && std::isfinite(c))
                {
                    double sum = c, wsum = 1.0;
                    for (int k = 0; k < r.count && k < (int)nb.size(); ++k)
                    {
                        const float w = r.weight[N * k + i];
                        if (w <= 0.f || !nb[k])
                            continue;
                        bool ok = false;
                        const float v = sampleBilinear(nb[k], nbStride[k], W, H, x + r.dx[N * k + i],
                                                       y + r.dy[N * k + i], ok);
                        if (!ok)
                            continue;
                        sum += (double)w * v;
                        wsum += w;
                    }
                    o = c + amount * ((float)(sum / wsum) - c);
                }
                out[(int64_t)y * outStride + x] = o;
            }
    });
}

} // namespace cleanpass
