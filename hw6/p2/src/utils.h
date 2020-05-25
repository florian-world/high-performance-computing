// Do not edit!
#pragma once

#include <cuda_runtime.h>
#include <curand.h>

/// Print the error message if a CUDA API execution failed.
void _cudaCheck(const char *file, int line, const char *func, cudaError_t code);

/// Print the error message if a CURAND API execution failed.
void _curandCheck(const char *file, int line, const char *func, curandStatus_t code);

// Wrap every cuda API with this check.
#define CUDA_CHECK(cmd) _cudaCheck(__FILE__, __LINE__, __PRETTY_FUNCTION__, (cmd))

// Wrap every curand API with this check.
#define CURAND_CHECK(cmd) _curandCheck(__FILE__, __LINE__, __PRETTY_FUNCTION__, (cmd))

/// Replace
///     kernel<<<blocks, threads>>>(arg1, ...);
/// with
///     CUDA_LAUNCH(kernel, blocks, threads, arg1, ...);
#define CUDA_LAUNCH(kernel, blocks, threads, ...) do {                           \
        cudaGetLastError();                                                      \
        kernel<<<(blocks), (threads)>>>(__VA_ARGS__);                            \
        _cudaCheck(__FILE__, __LINE__, __PRETTY_FUNCTION__, cudaGetLastError()); \
    } while (0)


/// Replace
///     kernel<<<blocks, threads, sharedMem, stream>>>(arg1, ...);
/// with
///     CUDA_LAUNCH_EX(kernel, blocks, threads, sharedMem, stream, arg1, ...);
#define CUDA_LAUNCH_EX(kernel, blocks, threads, sharedMem, stream, ...) do {     \
        cudaGetLastError();                                                      \
        kernel<<<(blocks), (threads), (sharedMem), (stream)>>>(__VA_ARGS__);     \
        _cudaCheck(__FILE__, __LINE__, __PRETTY_FUNCTION__, cudaGetLastError()); \
    } while (0)
