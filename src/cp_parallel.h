// CleanPass - tiny parallel-for shared by the core modules
// Copyright (c) 2026 Gabin Vidal. MIT License (see LICENSE).
#pragma once

#include <algorithm>
#include <thread>
#include <vector>

namespace cleanpass
{
namespace detail
{

inline int threadCount(int requested)
{
    int n = requested > 0 ? requested : (int)std::thread::hardware_concurrency();
    return std::max(1, std::min(n, 64));
}

// Runs fn(begin, end) on [0, n) split in contiguous chunks, one per thread.
template <class Fn>
void parallelFor(int n, int threads, Fn fn)
{
    if (n <= 0)
        return;
    threads = std::max(1, std::min(threads, n / 64 + 1));
    if (threads == 1)
    {
        fn(0, n);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads);
    const int chunk = (n + threads - 1) / threads;
    for (int t = 0; t < threads; ++t)
    {
        const int b = t * chunk, e = std::min(n, b + chunk);
        if (b >= e)
            break;
        pool.emplace_back([=]() { fn(b, e); });
    }
    for (auto& th : pool)
        th.join();
}

} // namespace detail
} // namespace cleanpass
