// CleanPass - core algorithm
// Copyright (c) 2026 Gabin Vidal. MIT License (see LICENSE).
//
// This file has no dependency on Nuke or OIDN, so the algorithm can be built
// and tested on its own. The Nuke node (CleanPass.cpp) is a thin wrapper
// around it, and the OIDN "expected value" predictor is injected as a callback.
//
// Overview
// --------
// A firefly is a pixel (or a tiny cluster) whose energy is far above what the
// surrounding samples say it should be. Unlike a denoiser, we do NOT smooth the
// image: we find those outliers and correct only them. Every other pixel is
// returned bit-for-bit, so the render's grain and fine detail are untouched.
//
//  1. Isolation  : the pixel must stand above the brightest part of a ring of
//                  neighbours (3rd-largest value, so 1 px lines are protected).
//  2. Expectation: it must also stand far above an "expected" value, either a
//                  guide-aware robust estimate (Statistical) or an OIDN
//                  prediction computed on a pre-cleaned copy of the frame.
//                  The threshold rises with the local noise level.
//  3. Temporal   : optional - a firefly does not survive to the next or the
//                  previous frame, a real highlight does.
//  4. Correction : replace / clamp only the detected pixels (+ their filter
//                  halo), and propagate the same correction to light AOVs so
//                  that the AOVs still add up to the corrected beauty.
//  5. Energy     : where the "fireflies" are dense, they are not accidents but
//                  the sampling noise of the light itself (indirect light at
//                  low sample counts): removing them would darken the image.
//                  Their energy is given back as a smooth field. Isolated
//                  fireflies are simply removed.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace cleanpass
{

enum class ReferenceMode : int
{
    OIDN = 0,        // ML prediction from Intel Open Image Denoise
    Statistical = 1  // guide-aware robust neighbourhood estimate (no OIDN)
};

enum class Method : int
{
    Reference = 0,   // replace with the expected value
    LocalMedian = 1, // replace with the per-channel median of the neighbours
    Clamp = 2        // scale down (hue kept) to the threshold level
};

struct Params
{
    // detection
    ReferenceMode reference = ReferenceMode::OIDN;
    float threshold = 1.5f;   // stops above the expected value
    float softness = 0.75f;   // stops, width of the soft transition
    float isolation = 0.75f;  // stops above the surrounding ring
    float adaptivity = 0.25f; // how much local noise raises the threshold
    float minLevel = 0.0f;    // ignore pixels darker than this (energy)
    int size = 2;             // max firefly radius in pixels (1..3), multi-scale
    int grow = 1;             // halo radius corrected around each firefly (0..3)
    bool fixNonFinite = true; // treat NaN / inf pixels as fireflies

    // correction
    Method method = Method::Reference;
    bool conserveEnergy = true; // give back the energy of dense noise (see compensateLayer)
    float energyRadius = 16.0f; // pixels (sigma of the smooth field)
    float energyAmount = 1.0f;  // 0..1

    // temporal
    bool temporal = true;     // on by default: protects real micro-highlights
    int motionRadius = 3; // search radius in the neighbour frames (pixels)

    // internal: how much more aggressive (stops) the pre-clean that feeds OIDN is
    float preCleanBias = 1.0f;

    int threads = 0; // 0 = hardware concurrency
};

// A read-only view on 3 channels, planar or interleaved.
struct View3
{
    const float* ch[3] = {nullptr, nullptr, nullptr};
    int64_t pixStride = 1;
    int64_t rowStride = 0;

    bool valid() const { return ch[0] != nullptr; }
    float at(int x, int y, int c) const { return ch[c][y * rowStride + x * pixStride]; }

    static View3 interleaved(const float* p, int width)
    {
        View3 v;
        v.ch[0] = p; v.ch[1] = p + 1; v.ch[2] = p + 2;
        v.pixStride = 3; v.rowStride = (int64_t)width * 3;
        return v;
    }
    static View3 planar(const float* r, const float* g, const float* b, int64_t rowStride)
    {
        View3 v;
        v.ch[0] = r; v.ch[1] = g; v.ch[2] = b;
        v.pixStride = 1; v.rowStride = rowStride;
        return v;
    }
};

struct Inputs
{
    int width = 0, height = 0;
    View3 beauty;       // required
    View3 albedo;       // optional guide
    View3 normal;       // optional guide
    View3 prev, next;   // optional, beauty at frame -1 / +1 (temporal)
    const float* mask = nullptr; // optional, w*h, 0..1
    int64_t maskRowStride = 0;
};

// Expected-value predictor (OIDN). color/out are interleaved RGB (w*h*3).
// Returns false and fills err on failure.
using ReferenceFn = std::function<bool(int width, int height, const float* color,
                                       const View3& albedo, const View3& normal,
                                       float* out, std::string& err)>;

using AbortFn = std::function<bool()>;

struct Analysis
{
    int width = 0, height = 0;
    bool valid = false;
    std::string warning;

    // dense, w*h: final firefly matte (mask applied, mix not applied)
    std::vector<float> matte;

    // per hit (corrected pixel)
    std::vector<int> hitIndex;      // y*width + x
    std::vector<float> hitM;        // matte value
    std::vector<float> hitDelta;    // 3/hit : amount removed from the beauty (finite)
    std::vector<float> hitValue;    // 3/hit : fully-corrected beauty value
    std::vector<float> hitIn;       // 3/hit : original beauty (sanitised)
    std::vector<float> hitMed;      // 3/hit : per-channel ring median of the beauty
    std::vector<uint8_t> hitFlags;  // bit0: non-finite input, bit1: core (not halo)
    std::vector<float> hitKeep;     // share of the removed energy given back (dense noise, see compensateLayer)
    int numCore = 0;

    // AOVs
    struct Aov
    {
        std::vector<float> delta;  // 3/hit
        std::vector<float> med;    // 3/hit  (fallback for non-finite AOV pixels)
        std::vector<float> val;    // 3/hit  (scratch)
        std::vector<float> excess; // 3/hit  (scratch)
    };
    std::vector<Aov> aovs;
    bool aovsFinalized = false;

    // statistics
    double seconds = 0.0;
};

// Runs detection and computes the beauty correction. Thread-safe as long as
// each call gets its own Analysis. `reference` may be empty if Statistical.
bool analyze(const Inputs& in, const Params& p, const ReferenceFn& reference,
             Analysis& out, std::string& err, const AbortFn& aborted = AbortFn());

// AOV propagation: call beginAovs(), then addAovChannel() for every (layer, rgb
// component) with planar data covering the analysis box, then finalizeAovs().
void beginAovs(Analysis& a, int numLayers);
void addAovChannel(Analysis& a, const Params& p, int layer, int comp,
                   const float* data, int64_t rowStride);
void finalizeAovs(Analysis& a);

// Applies the correction in place on an output buffer that already holds the
// input values. `layer` = -1 for the beauty, else the AOV index.
// out[c] (c = 0..2, may be null) point at the pixel that corresponds to the
// analysis-box pixel (offX, offY); the buffer is outW x outH with rowStride.
// Coordinates outside the buffer are skipped.
struct OutBuffer
{
    float* ch[3] = {nullptr, nullptr, nullptr};
    int x0 = 0, y0 = 0;   // analysis-box coordinate of the buffer's first pixel
    int width = 0, height = 0;
    int64_t rowStride = 0;
    int64_t pixStride = 1;
};
void applyLayer(const Analysis& a, const Params& p, int layer, const OutBuffer& out, float mix);

// Energy of dense noise: adds back, as a smooth field, the energy that the
// correction removed where the detections are dense and carry a real part of
// the local light (hitKeep): out += amount * G(keep * removed) * albedo /
// G(albedo), per channel. Isolated fireflies are not given back. Linear in the removed
// values, so AOVs that add up to the beauty keep adding up. Call after
// applyLayer, with a buffer that covers the whole analysis box. `albedo`
// (analysis box, optional) puts the energy back where the surfaces reflect it.
void compensateLayer(const Analysis& a, const Params& p, int layer, const OutBuffer& out, const View3& albedo,
                     float mix, int threads = 0);

// Writes the matte (times mix) into a single-channel buffer that is assumed to
// be zero-initialised. Uses ch[0] of `out`.
void writeMatte(const Analysis& a, const OutBuffer& out, float mix);

// Small helpers shared with the wrapper.
inline float energyOf(float r, float g, float b)
{
    float e = r > g ? r : g;
    e = e > b ? e : b;
    return e > 0.f ? e : 0.f;
}

} // namespace cleanpass
