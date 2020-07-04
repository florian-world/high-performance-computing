#pragma once

#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

template <typename Func>
double benchmark(int N, Func func) {
    double dt = 0.0;  // Time per invocation in seconds.

    // TODO: (OPTIONAL) Implement the measurement procedure
    // here and use this function for all your measurements.

    int warmup = 1 + N / 10;
    for (int i = 0; i < warmup; ++i) {
        func();
    }
    cudaDeviceSynchronize();

    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < N; ++i)
        func();

    cudaDeviceSynchronize();

    auto t1 = std::chrono::steady_clock::now();

    double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    dt = ns / N * 1e-9;

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
