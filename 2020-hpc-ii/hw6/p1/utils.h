#pragma once

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

/*
 * Benchmark the given function `N` times and return the average execution time.
 */
template <typename Func>
double benchmark(int N, Func func) {
    for (int i = 0; i < N / 10 + 1; ++i)
        func();  // Warmup.
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i)
        func();
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    auto ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    double dt = 1e-9 * ns / N;
    return dt;
}

/// Print the error message if a CUDA API execution failed.
void _cudaCheck(const char *file, int line, const char *func, cudaError_t code) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d in %s:%s\n",
                file, line, func, cudaGetErrorString(code));
        exit(1);
    }
}

// Wrap every cuda API with this check.
#define CUDA_CHECK(cmd) _cudaCheck(__FILE__, __LINE__, __PRETTY_FUNCTION__, (cmd))

/// Replace
///     kernel<<<blocks, threads>>>(arg1, ...);
/// with
///     CUDA_LAUNCH(kernel, blocks, threads, arg1, ...);
#define CUDA_LAUNCH(kernel, blocks, threads, ...) do {                           \
        cudaGetLastError();                                                      \
        kernel<<<(blocks), (threads)>>>(__VA_ARGS__);                            \
        _cudaCheck(__FILE__, __LINE__, __PRETTY_FUNCTION__, cudaGetLastError()); \
    } while (0)

