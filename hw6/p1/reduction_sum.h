#pragma once

#include "utils.h"
#include <algorithm>
#include <numeric>

constexpr int kWarpSize = 32;


/// Kernel for testing `sumWarp`. Do not edit.
__global__ void sumWarpTestKernel(const double *a, double *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // All threads of a warp must call `sumWarp`!
    double sum = sumWarp(idx < N ? a[idx] : 0.0);
    if (threadIdx.x % warpSize == 0 && idx / warpSize < (N + warpSize - 1) / warpSize)
        b[idx / warpSize] = sum;
}

/// Check results of `sumWarp`. Do not edit.
inline void sumWarpCheck(int N, int K, const double *aHost, const double *bHost) {
    for (int k = 0; k < K; ++k) {
        double expected = std::accumulate(aHost + k * kWarpSize, aHost + std::min((k + 1) * kWarpSize, N), 0.0);
        double received = bHost[k];
        if (expected != received) {
            printf("sumWarp incorrect result:  k=%d  expected=%f  received=%f\n", k, expected, received);
            exit(1);
        }
    }
}

/*
/// Kernel for testing `sumWarpAll`. Do not edit.
__global__ void sumWarpAllTestKernel(const double *a, double *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // All threads of a warp must call `sumWarpAll`!
    double sum = sumWarpAll(idx < N ? a[idx] : 0.0);
    if (idx < N)
        b[idx] = sum;
}

/// Check result of `sumWarpAll`. Do not edit.
inline void sumWarpAllCheck(int N, int K, const double *aHost, const double *bHost) {
    for (int k = 0; k < K; k += kWarpSize) {
        double expected = std::accumulate(aHost + k, aHost + std::min(k + kWarpSize, N), 0.0);
        for (int j = k; j < std::min(k + kWarpSize, N); ++j) {
            double received = bHost[j];
            if (expected != received) {
                printf("sumWarpAll incorrect result:  k=%d j=%d  expected=%f  received=%f\n", k, j, expected, received);
                exit(1);
            }
        }
    }
}
*/

// Do not edit.
__global__ void sumBlockTestKernel(const double *a, double *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = sumBlock(idx < N ? a[idx] : 0.0);
    if (threadIdx.x == 0)
        b[blockIdx.x] = sum;
}

// Do not edit.
inline void sumBlockCheck(int N, int K, const double *aHost, const double *bHost) {
    for (int k = 0; k < K; ++k) {
        double expected = std::accumulate(aHost + k * 1024, aHost + std::min((k + 1) * 1024, N), 0.0);
        double received = bHost[k];
        if (expected != received) {
            printf("sumBlock incorrect result:  k=%d  expected=%f  received=%f\n", k, expected, received);
            exit(1);
        }
    }
}



/// Test small reductions (warp-level and block-level).
template <typename Kernel, typename CheckFunc>
void testSmallSum(Kernel kernel, CheckFunc checkFunc, int div, int N) {
    int K = (N + div - 1) / div;
    double *aHost;
    double *bHost;
    double *aDev;
    double *bDev;

    CUDA_CHECK(cudaMallocHost(&aHost, N * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&bHost, K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&aDev, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bDev, K * sizeof(double)));

    for (int i = 0; i < N; ++i)
        aHost[i] = i;
    CUDA_CHECK(cudaMemcpy(aDev, aHost, N * sizeof(double), cudaMemcpyHostToDevice));

    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;
    kernel<<<blocks, threads>>>(aDev, bDev, N);
    CUDA_CHECK(cudaMemcpy(bHost, bDev, K * sizeof(double), cudaMemcpyDeviceToHost));

    checkFunc(N, K, aHost, bHost);

    CUDA_CHECK(cudaFree(bDev));
    CUDA_CHECK(cudaFree(aDev));
    CUDA_CHECK(cudaFreeHost(bHost));
    CUDA_CHECK(cudaFreeHost(aHost));
}

/// Test large reductions (up to 1024^3 and larger).
template <typename Func>
void testLargeSum(const char *name, Func func, int N) {
    double *aHost;
    double *aDev;
    double *bDev;

    CUDA_CHECK(cudaMallocHost(&aHost, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&aDev, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bDev, 1 * sizeof(double)));

    for (int i = 0; i < N; ++i)
        aHost[i] = i % 12345678;
    CUDA_CHECK(cudaMemcpy(aDev, aHost, N * sizeof(double), cudaMemcpyHostToDevice));

    func(aDev, bDev, N);

    double expected = std::accumulate(aHost, aHost + N, 0.0);
    double received;
    CUDA_CHECK(cudaMemcpy(&received, bDev, 1 * sizeof(double), cudaMemcpyDeviceToHost));
    if (expected != received) {
        printf("large %s incorrect result:  N=%d  expected=%f  received=%f\n", name, N, expected, received);
        exit(1);
    }

    CUDA_CHECK(cudaFree(bDev));
    CUDA_CHECK(cudaFree(aDev));
    CUDA_CHECK(cudaFreeHost(aHost));
}
