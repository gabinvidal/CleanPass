// CleanPass - low-pass helpers
// Copyright (c) 2026 Gabin Vidal. MIT License (see LICENSE).
//
// Smooth (Gaussian) fields computed on a coarse grid: the image is summed in
// f x f cells, blurred there, and read back with a bilinear lookup. With a
// sigma of 8 - 16 px this costs a fraction of a full-resolution blur, and the
// result is just as smooth. Everything here is linear in the data.
//
// Used to give back, as a smooth field, the energy that a robust local
// operation removed: the firefly stage on dense noise, the deflicker where its
// outlier tests prefer the darker samples.

#pragma once

#include "cp_parallel.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cleanpass
{
namespace lp
{

struct Grid
{
    int W = 0, H = 0; // full resolution
    int f = 1;        // cell size
    int cw = 0, ch = 0;
    float sigmaC = 1.f; // blur sigma, in cells
};

inline Grid grid(int W, int H, float sigma)
{
    Grid g;
    g.W = W;
    g.H = H;
    sigma = std::max(1.f, sigma);
    g.f = std::max(1, (int)std::lround(sigma / 4.f));
    g.cw = (W + g.f - 1) / g.f;
    g.ch = (H + g.f - 1) / g.f;
    g.sigmaC = sigma / (float)g.f;
    return g;
}

// Sums value(x, y) over each cell. value must be thread-safe.
template <class Fn>
void down(const Grid& g, Fn value, std::vector<float>& out, int nth)
{
    out.assign((size_t)g.cw * g.ch, 0.f);
    detail::parallelFor(g.ch, nth, [&](int c0, int c1) {
        for (int cy = c0; cy < c1; ++cy)
            for (int y = cy * g.f; y < std::min(g.H, (cy + 1) * g.f); ++y)
                for (int x = 0; x < g.W; ++x)
                    out[(size_t)cy * g.cw + x / g.f] += value(x, y);
    });
}

// Separable Gaussian on the coarse grid (zero outside the frame).
inline void blur(const Grid& g, std::vector<float>& a, int nth)
{
    const int r = std::max(1, (int)std::ceil(3.f * g.sigmaC));
    std::vector<float> k(2 * r + 1);
    float s = 0.f;
    for (int i = -r; i <= r; ++i)
        s += k[i + r] = std::exp(-0.5f * (float)(i * i) / (g.sigmaC * g.sigmaC));
    for (float& v : k)
        v /= s;
    std::vector<float> t(a.size(), 0.f);
    detail::parallelFor(g.ch, nth, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < g.cw; ++x)
            {
                float acc = 0.f;
                for (int i = std::max(-r, -x); i <= std::min(r, g.cw - 1 - x); ++i)
                    acc += k[i + r] * a[(size_t)y * g.cw + x + i];
                t[(size_t)y * g.cw + x] = acc;
            }
    });
    detail::parallelFor(g.ch, nth, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            for (int x = 0; x < g.cw; ++x)
            {
                float acc = 0.f;
                for (int i = std::max(-r, -y); i <= std::min(r, g.ch - 1 - y); ++i)
                    acc += k[i + r] * t[(size_t)(y + i) * g.cw + x];
                a[(size_t)y * g.cw + x] = acc;
            }
    });
}

// Bilinear lookup at a full-resolution position (pixel centres at integers).
inline float at(const Grid& g, const std::vector<float>& a, float x, float y)
{
    const float u = std::max(0.f, std::min((float)g.cw - 1.f, (x + 0.5f) / (float)g.f - 0.5f));
    const float v = std::max(0.f, std::min((float)g.ch - 1.f, (y + 0.5f) / (float)g.f - 0.5f));
    const int x0 = std::min(g.cw - 1, (int)u), y0 = std::min(g.ch - 1, (int)v);
    const int x1 = std::min(g.cw - 1, x0 + 1), y1 = std::min(g.ch - 1, y0 + 1);
    const float fx = u - (float)x0, fy = v - (float)y0;
    const float* r0 = &a[(size_t)y0 * g.cw];
    const float* r1 = &a[(size_t)y1 * g.cw];
    return (1.f - fy) * ((1.f - fx) * r0[x0] + fx * r0[x1]) + fy * ((1.f - fx) * r1[x0] + fx * r1[x1]);
}

// Per-pixel density (value per full-resolution pixel) of a blurred cell sum.
inline float density(const Grid& g, const std::vector<float>& a, float x, float y)
{
    return at(g, a, x, y) / (float)(g.f * g.f);
}

inline float smoothstep(float a, float b, float x)
{
    if (b <= a)
        return x >= b ? 1.f : 0.f;
    const float t = std::max(0.f, std::min(1.f, (x - a) / (b - a)));
    return t * t * (3.f - 2.f * t);
}

} // namespace lp
} // namespace cleanpass
