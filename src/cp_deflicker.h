// CleanPass - Deflicker (motion-compensated temporal stabilisation)
// Copyright (c) 2026 Gabin Vidal. MIT License (see LICENSE).
//
// Flicker in CG renders is the sampling noise and aliasing that changes from
// one frame to the next: shimmering speculars, crawling sub-pixel detail,
// boiling GI. A spatial filter cannot remove it without blurring. Here every
// pixel of frame t is matched (block matching on the beauty and, when present,
// on the noise-free albedo / normal guides) in the neighbouring frames, and
// averaged with the matched samples that really look like the same surface.
// Where the match is poor (occlusion, real change), the neighbour gets no
// weight, so nothing ghosts.
//
// The result is a set of per-pixel offsets and weights, applied identically
// to the beauty and to every AOV, so light AOVs keep adding up to the beauty.

#pragma once

#include "cp_core.h"

#include <cstdint>
#include <vector>

namespace cleanpass
{

struct DeflickerParams
{
    int frames = 1;          // temporal radius: 1 = t-1, t+1 ; 2 = also t-2, t+2
    int search = 4;          // motion search radius in pixels (per frame of distance)
    int patch = 2;           // matching patch radius (2 = 5x5)
    float tolerance = 2.0f;  // how different a match may be and still be averaged
    float amount = 1.0f;     // 0..1 mix
    bool useGuides = true;   // use albedo / normal for the matching when available
    bool useVectors = true;  // use the renderer motion vectors of the current frame when present
    bool useDepth = true;    // reject samples from another depth (disocclusion) when depth is present
    bool useVariance = true; // per-pixel noise level from a variance pass when present
    int threads = 0;
};

struct DeflickerFrame
{
    View3 beauty;            // required
    View3 albedo, normal;    // optional
    View3 depth;             // optional (channel 0): camera depth, any unit
    View3 motion;            // optional, current frame: renderer motion vectors (channels 0 = x, 1 = y).
                             // Any common convention (pixels or NDC, forward or backward, y up or
                             // down, full or half frame): it is calibrated against the image.
    View3 variance;          // optional, current frame: per-channel variance of the beauty
    View3 raw;               // optional, current frame: the beauty before any cleaning (firefly stage),
                             // so that the change test compares like with like (neighbours are raw)
    int offset = 0;          // frame distance (-2, -1, +1, +2)
};

struct DeflickerResult
{
    int width = 0, height = 0;
    int count = 0;                 // number of neighbour frames
    std::vector<float> dx, dy;     // count * w * h : matched (sub-pixel) offset in each neighbour
    std::vector<float> weight;     // count * w * h : weight of each neighbour sample
    std::vector<float> geo;        // count * w * h : geometric part of the weight only (inside the
                                   // frame, same surface by the guides, same depth) - no noise test
    std::vector<float> same;       // count * w * h : 1 = the neighbour shows the same light (no
                                   // significant change of the local mean, any channel), 0 = a real change
    float noise = 0.f;             // estimated matching noise (diagnostics)
    // motion vectors: displacement to the frame at offset o = o * (vecX * mx, vecY * my)
    bool vectorsUsed = false;
    float vecX = 0.f, vecY = 0.f;
    float vectorShare = 0.f;       // fraction of the samples whose match comes from the vectors
    bool depthUsed = false, varianceUsed = false;
    // fixed render seed: share of the pixels identical to a neighbour frame
    float identical = 0.f;
    float noiseCorrelation = 0.f;  // high-frequency correlation with the first neighbour on flat areas
    bool fixedSeed = false;
    double seconds = 0.0;
    bool valid = false;
};

// `cur` is frame t (ideally already firefly-cleaned). Neighbours whose beauty
// is invalid are ignored. Returns false only on abort.
bool deflickerAnalyze(int width, int height, const DeflickerFrame& cur, const std::vector<DeflickerFrame>& neighbours,
                      const DeflickerParams& p, DeflickerResult& out, const AbortFn& aborted = AbortFn());

// Refines an offset field (w*h, from deflickerAnalyze) between two clean
// images - e.g. two denoised frames - by a +-1 px search and a sub-pixel fit.
// Noise-free images give a much more precise alignment than the noisy ones.
void deflickerRefine(int width, int height, const DeflickerFrame& cur, const DeflickerFrame& nb, float* dx, float* dy,
                     int threads = 0);

// Applies the temporal average to one channel. All buffers are w x h with the
// given row strides (x-contiguous). nb[k] may be null (that neighbour is skipped).
void deflickerApplyChannel(const DeflickerResult& r, float amount, const float* cur, int64_t curStride,
                           const std::vector<const float*>& nb, const std::vector<int64_t>& nbStride, float* out,
                           int64_t outStride);

} // namespace cleanpass
