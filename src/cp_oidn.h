// CleanPass - OIDN expected-value predictor
// Copyright (c) 2026 Gabin Vidal. MIT License (see LICENSE).
//
// OIDN is used here as a *predictor*, not as the output: it tells the core
// what each pixel is expected to look like given its neighbourhood and the
// albedo / normal guides. The result never reaches the image directly.

#pragma once

#include "cp_core.h"

#include <functional>
#include <string>

namespace cleanpass
{

struct OidnSettings
{
    int device = 0;          // 0 = CPU, 1 = CUDA (falls back to CPU)
    int quality = 1;         // 0 = Balanced, 1 = High, 2 = Fast (OIDN 2.2+)
    bool prefilterAux = false; // guides are noisy: let OIDN clean them first
    int maxMemoryMB = 0;     // 0 = no limit
};

// Runs the OIDN "RT" filter (HDR) on `color` (interleaved RGB, w*h*3) into
// `out`. Guides are optional. Thread-safe: runs are serialised on a single
// process-wide device. `aborted` is polled by OIDN's progress monitor.
bool oidnPredict(const OidnSettings& s, int width, int height, const float* color,
                 const View3& albedo, const View3& normal, float* out,
                 std::string& err, const std::function<bool()>& aborted);

// Name of the device actually in use ("CPU", "CUDA"), or "" before first use.
const char* oidnActiveDevice();

// Releases the shared device (call at shutdown).
void oidnRelease();

} // namespace cleanpass
