// CleanPass - OIDN expected-value predictor
// Copyright (c) 2026 Gabin Vidal. MIT License (see LICENSE).

#include "cp_oidn.h"

#include <OpenImageDenoise/oidn.hpp>

#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

#ifdef _WIN32
// SRWLOCK is statically initialised: no constructor has to run when the DLL is
// loaded with LoadLibrary inside Nuke. (std::mutex built with a recent MSVC
// can crash against the older msvcp140.dll that ships with Nuke.)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
namespace
{
SRWLOCK g_lock = SRWLOCK_INIT;
struct Guard
{
    Guard() { AcquireSRWLockExclusive(&g_lock); }
    ~Guard() { ReleaseSRWLockExclusive(&g_lock); }
};
} // namespace
#else
#include <mutex>
namespace
{
std::mutex g_lock;
struct Guard
{
    Guard() { g_lock.lock(); }
    ~Guard() { g_lock.unlock(); }
};
} // namespace
#endif

namespace cleanpass
{
namespace
{

oidn::DeviceRef g_device;
int g_deviceType = -1;      // device actually created (0 CPU, 1 CUDA)
int g_requestedType = -1;   // device that was asked for
const char* g_deviceName = "";

bool ensureDevice(int requested, std::string& err)
{
    if (g_device && g_requestedType == requested)
        return true;
    g_device = nullptr;
    g_deviceType = -1;
    g_requestedType = requested;
    g_deviceName = "";

    const oidn::DeviceType types[2] = {oidn::DeviceType::CPU, oidn::DeviceType::CUDA};
    int tryType = requested == 1 ? 1 : 0;
    for (int attempt = 0; attempt < 2; ++attempt)
    {
        try
        {
            oidn::DeviceRef dev = oidn::newDevice(types[tryType]);
            const char* msg = nullptr;
            if (!dev || dev.getError(msg) != oidn::Error::None)
                throw std::runtime_error(msg ? msg : "cannot create device");
            dev.commit();
            if (dev.getError(msg) != oidn::Error::None)
                throw std::runtime_error(msg ? msg : "cannot commit device");
            g_device = dev;
            g_deviceType = tryType;
            g_deviceName = tryType == 1 ? "CUDA" : "CPU";
            if (tryType != requested)
                std::cerr << "[CleanPass] CUDA unavailable, using CPU for OIDN" << std::endl;
            return true;
        }
        catch (const std::exception& e)
        {
            err = e.what();
            if (tryType == 0)
                break;
            tryType = 0; // fall back to CPU
        }
    }
    g_requestedType = -1;
    return false;
}

void pack(const View3& v, int w, int h, std::vector<float>& dst, bool normal)
{
    dst.resize((size_t)w * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c)
            {
                float f = v.at(x, y, c);
                if (!std::isfinite(f))
                    f = 0.f;
                if (normal)
                    f = f < -1.f ? -1.f : (f > 1.f ? 1.f : f);
                else
                    f = f < 0.f ? 0.f : (f > 1.f ? 1.f : f);
                dst[((size_t)y * w + x) * 3 + c] = f;
            }
}

struct ProgressCtx
{
    const std::function<bool()>* aborted;
};

bool progressCb(void* user, double)
{
    const ProgressCtx* ctx = static_cast<const ProgressCtx*>(user);
    return !(ctx->aborted && *ctx->aborted && (*ctx->aborted)());
}

} // namespace

bool oidnPredict(const OidnSettings& s, int width, int height, const float* color,
                 const View3& albedo, const View3& normal, float* out,
                 std::string& err, const std::function<bool()>& aborted)
{
    if (width <= 0 || height <= 0)
    {
        err = "empty image";
        return false;
    }
    // Guides are packed outside the lock (pure CPU work).
    std::vector<float> alb, nrm;
    const bool useAlb = albedo.valid();
    const bool useNrm = normal.valid() && useAlb; // OIDN needs albedo to use normals
    if (useAlb)
        pack(albedo, width, height, alb, false);
    if (useNrm)
        pack(normal, width, height, nrm, true);

    Guard lock;
    if (!ensureDevice(s.device, err))
        return false;

    const size_t bytes = (size_t)width * height * 3 * sizeof(float);
    try
    {
        oidn::BufferRef bColor = g_device.newBuffer(bytes);
        oidn::BufferRef bOut = g_device.newBuffer(bytes);
        oidn::BufferRef bAlb, bNrm;
        bColor.write(0, bytes, color);
        if (useAlb)
        {
            bAlb = g_device.newBuffer(bytes);
            bAlb.write(0, bytes, alb.data());
        }
        if (useNrm)
        {
            bNrm = g_device.newBuffer(bytes);
            bNrm.write(0, bytes, nrm.data());
        }

        oidn::FilterRef f = g_device.newFilter("RT");
        f.setImage("color", bColor, oidn::Format::Float3, width, height);
        f.setImage("output", bOut, oidn::Format::Float3, width, height);
        if (useAlb)
            f.setImage("albedo", bAlb, oidn::Format::Float3, width, height);
        if (useNrm)
            f.setImage("normal", bNrm, oidn::Format::Float3, width, height);
        f.set("hdr", true);
        f.set("cleanAux", !s.prefilterAux);
        oidn::Quality q = s.quality == 1 ? oidn::Quality::High : oidn::Quality::Balanced;
#if defined(OIDN_VERSION) && OIDN_VERSION >= 20200
        if (s.quality == 2)
            q = oidn::Quality::Fast;
#endif
        f.set("quality", q);
        if (s.maxMemoryMB > 0)
            f.set("maxMemoryMB", s.maxMemoryMB);
        ProgressCtx ctx{&aborted};
        f.setProgressMonitorFunction(progressCb, &ctx);
        f.commit();

        const char* msg = nullptr;
        oidn::Error e = g_device.getError(msg);
        if (e == oidn::Error::None)
        {
            f.execute();
            e = g_device.getError(msg);
        }
        if (e != oidn::Error::None)
        {
            err = e == oidn::Error::Cancelled ? "cancelled" : (msg ? msg : "OIDN error");
            return false;
        }
        bOut.read(0, bytes, out);
    }
    catch (const std::exception& ex)
    {
        const char* drained = nullptr;
        g_device.getError(drained); // do not leak the error into the next run
        err = ex.what();
        return false;
    }
    return true;
}

const char* oidnActiveDevice()
{
    Guard lock;
    return g_deviceName;
}

void oidnRelease()
{
    Guard lock;
    g_device = nullptr;
    g_deviceType = -1;
    g_requestedType = -1;
    g_deviceName = "";
}

} // namespace cleanpass
