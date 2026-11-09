// CleanPass - tiny lock usable from a Nuke plugin DLL
// Copyright (c) 2026 Gabin Vidal. MIT License (see LICENSE).
//
// A plain std::atomic_flag lock. We avoid std::mutex on purpose: a std::mutex
// compiled with a recent MSVC can crash when the process has already loaded
// the older msvcp140.dll that ships with Nuke. atomic_flag is header-only.
// The lock only guards the per-node analysis cache, so waiting threads simply
// sleep while the first one computes the frame.

#pragma once

#include <atomic>
#include <chrono>
#include <thread>

namespace cleanpass
{
class Lock
{
public:
    Lock() = default;
    Lock(const Lock&) = delete;
    Lock& operator=(const Lock&) = delete;

    void lock()
    {
        int spins = 0;
        while (m_flag.test_and_set(std::memory_order_acquire))
        {
            if (++spins < 64)
                std::this_thread::yield();
            else
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    void unlock() { m_flag.clear(std::memory_order_release); }

private:
    std::atomic_flag m_flag = ATOMIC_FLAG_INIT;
};

class LockGuard
{
public:
    explicit LockGuard(Lock& l) : m_l(l) { m_l.lock(); }
    ~LockGuard() { m_l.unlock(); }
    LockGuard(const LockGuard&) = delete;
    LockGuard& operator=(const LockGuard&) = delete;

private:
    Lock& m_l;
};
} // namespace cleanpass
