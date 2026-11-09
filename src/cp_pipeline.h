// CleanPass - the three cleaning stages chained together
// Copyright (c) 2026 Gabin Vidal. MIT License (see LICENSE).
//
//   1. Firefly   - removes outliers only (cp_core)
//   2. Deflicker - motion-compensated temporal stabilisation (cp_deflicker)
//   3. Denoise   - Intel Open Image Denoise on the stabilised frame
//
// Every stage can be switched off. No Nuke dependency, so the whole chain can
// be tested outside Nuke.

#pragma once

#include "cp_core.h"
#include "cp_deflicker.h"

#include <array>
#include <string>
#include <memory>
#include <vector>

namespace cleanpass
{

struct PipelineParams
{
    bool firefly = true;
    Params ff;
    bool deflicker = false;
    DeflickerParams df;
    bool denoise = false;
    float denoiseAmount = 1.f;
    // Temporal stabilisation of the denoised result: frames t-1 and t+1 go
    // through the same chain, and their denoised results are blended in with
    // the motion of the deflicker stage where they agree. Removes the
    // frame-to-frame "boiling" a per-frame denoiser leaves.
    bool stabilize = false;
    float stabilizeTolerance = 0.04f; // max difference (display-like units) still blended
    // AOVs (see pipelineAovs)
    bool denoiseAovs = false;
    bool aovsMatchBeauty = true;      // complete light-group sets keep adding up to the beauty
    // internal: run the deflicker analysis (1 frame) even when nothing needs it
    // here, so that the result can later serve as `precomputed`
    bool keepAnalysis = false;
};

struct PipelineFrame
{
    View3 beauty, albedo, normal;
    View3 depth;    // optional: camera depth (channel 0)
    View3 motion;   // optional: renderer motion vectors (channels 0, 1), used on the current frame
    View3 variance; // optional: variance of the beauty, used on the current frame
    int offset = 0; // frame distance from the current frame
};

struct PipelineInput
{
    int width = 0, height = 0;
    PipelineFrame cur;
    std::vector<PipelineFrame> neighbours; // any of -2, -1, +1, +2 (missing ones simply absent)
    const float* mask = nullptr;           // w*h, 0..1, limits the firefly stage
    int64_t maskStride = 0;
    // stabilisation: already known denoised results of frames t-1 / t+1 (planar
    // 3*w*h, e.g. from a cache), computed otherwise
    const std::vector<float>* denoisedPrev = nullptr;
    const std::vector<float>* denoisedNext = nullptr;
    // this frame already went through the firefly / deflicker / denoise stages
    // with the same chain (typically as the t+1 of the previous frame's
    // stabilisation): only the stabilisation is left to do. Must come from a
    // run without stabilisation and with deflicker frames <= 1.
    const struct PipelineResult* precomputed = nullptr;
};

struct PipelineResult
{
    int width = 0, height = 0;
    Analysis firefly;               // always valid (empty when the stage is off)
    DeflickerResult deflicker;      // valid when the stage ran
    std::vector<int> dfNeighbour;   // deflicker neighbour k -> index in PipelineInput::neighbours
    std::vector<float> beauty;      // 3 * w * h planar: final cleaned beauty
    std::vector<float> beautyStage2;   // 3 * w * h: after firefly + deflicker (kept when denoising)
    std::vector<float> beautyDenoised; // 3 * w * h: denoised, before stabilisation (kept when stabilising)
    std::vector<int> dfOffset;      // frame offset of each deflicker neighbour k
    // stabilisation: complete results computed for t-1 / t+1 (their `beauty`
    // is the denoised image), reusable as `precomputed` when that frame comes
    std::shared_ptr<PipelineResult> subPrev, subNext;
    bool reused = false; // stages 1 - 3 came from `precomputed`
    bool stabilized = false;
    std::string warning;
    double seconds[3] = {0, 0, 0};  // time of each stage
    bool valid = false;
};

// `reference` = OIDN predictor for the firefly stage, `denoise` = OIDN denoiser
// (same signature: interleaved colour in, interleaved colour out).
bool runPipeline(const PipelineInput& in, const PipelineParams& p, const ReferenceFn& reference,
                 const ReferenceFn& denoise, PipelineResult& out, std::string& err,
                 const AbortFn& aborted = AbortFn());

// Applies the firefly correction (AOV layer `layer`, see cp_core AOV API) and
// the deflicker to one AOV. cur: 3 planar channels w*h at frame t. nb: for each
// entry of PipelineInput::neighbours, the same AOV at that frame (or nulls).
// out: 3 planar channels w*h. albedo (optional): the beauty's albedo guide, where
// the energy of dense noise is given back. Denoise is not applied here (see
// pipelineAovs).
void pipelineAov(const PipelineResult& r, const PipelineParams& p, int layer, const std::array<const float*, 3>& cur,
                 const std::vector<std::array<const float*, 3>>& nb, const std::array<float*, 3>& out,
                 const View3& albedo = View3());

// Every AOV at once: firefly + deflicker (pipelineAov), then, with denoiseAovs,
// OIDN on each AOV with the beauty's guides. When the AOVs add up to the beauty
// (a complete set of light groups, checked on the image) and aovsMatchBeauty is
// on, the difference between the cleaned beauty and the sum of the cleaned AOVs
// is shared between them, so they add up exactly again. Otherwise only the
// stabilisation applied to the beauty is passed on, in proportion. `layers`:
// firefly AOV index of each AOV (-2 = none). Returns a warning (empty if none).
std::string pipelineAovs(const PipelineResult& r, const PipelineParams& p, const PipelineInput& in,
                         const ReferenceFn& denoise, const std::vector<int>& layers,
                         const std::vector<std::array<const float*, 3>>& cur,
                         const std::vector<std::vector<std::array<const float*, 3>>>& nb,
                         const std::vector<std::array<float*, 3>>& out);

} // namespace cleanpass
