#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>

constexpr int kWarpSize = 32;

// Kernel for testing `argMaxWarp`. Do not edit.
__global__ void argMaxWarpTestKernel(const double *a, Pair *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // All threads of a warp must call `argMaxWarp`!
    Pair argMax = argMaxWarp(idx < N ? a[idx] : 0.0);
    if (threadIdx.x % warpSize == 0 && idx / warpSize < (N + warpSize - 1) / warpSize)
        b[idx / warpSize] = argMax;
}

/// Check results of `argMaxWarp`. Do not edit.
void argMaxWarpCheck(int N, int K, const double *aHost, const Pair *bHost) {
    for (int k = 0; k < K; ++k) {
        int expectedIdx = std::max_element(aHost + k * kWarpSize, aHost + std::min((k + 1) * kWarpSize, N))
                          - (aHost + k * kWarpSize);
        Pair expected{aHost[k * kWarpSize + expectedIdx], expectedIdx};
        Pair received = bHost[k];
        if (expected.idx != received.idx || expected.max != received.max) {
            printf("argMaxWarp incorrect result:  k=%d  expected=%d %f  received=%d %f\n",
                   k, expected.idx, expected.max, received.idx, received.max);
            exit(1);
        }
    }
}

// Do not edit.
__global__ void argMaxBlockTestKernel(const double *a, Pair *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    Pair out = argMaxBlock(idx < N ? a[idx] : 0.0);
    if (threadIdx.x == 0)
        b[blockIdx.x] = out;
}


// Do not edit.
void argMaxBlockCheck(int N, int K, const double *aHost, const Pair *bHost) {
    for (int k = 0; k < K; ++k) {
        int expectedIdx = std::max_element(aHost + k * 1024, aHost + std::min((k + 1) * 1024, N))
                        - (aHost + k * 1024);
        Pair expected{aHost[k * 1024 + expectedIdx], expectedIdx};
        Pair received = bHost[k];
        if (expected.idx != received.idx || expected.max != received.max) {
            printf("argMaxBlock incorrect result:  k=%d  expected=%d %f  received=%d %f\n",
                   k, expected.idx, expected.max, received.idx, received.max);
            exit(1);
        }
    }
}


/// Test small argmax reductions (warp-level and block-level).
template <typename Kernel, typename CheckFunc>
void testSmallArgMax(Kernel kernel, CheckFunc checkFunc, int div, int N) {
    int K = (N + div - 1) / div;
    double *aHost;
    Pair *bHost;
    double *aDev;
    Pair *bDev;

    CUDA_CHECK(cudaMallocHost(&aHost, N * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&bHost, K * sizeof(Pair)));
    CUDA_CHECK(cudaMalloc(&aDev, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bDev, K * sizeof(Pair)));

    for (int i = 0; i < N; ++i) {
        // aHost[i] = (long long)i * i % 12345;
        aHost[i] = 10 * i;
    }
    std::mt19937 gen;
    std::shuffle(aHost, aHost + N, gen);
    CUDA_CHECK(cudaMemcpy(aDev, aHost, N * sizeof(double), cudaMemcpyHostToDevice));

    const int threads = 1024;
    const int blocks = (N + threads - 1) / threads;
    kernel<<<blocks, threads>>>(aDev, bDev, N);
    CUDA_CHECK(cudaMemcpy(bHost, bDev, K * sizeof(Pair), cudaMemcpyDeviceToHost));

    checkFunc(N, K, aHost, bHost);

    CUDA_CHECK(cudaFree(bDev));
    CUDA_CHECK(cudaFree(aDev));
    CUDA_CHECK(cudaFreeHost(bHost));
    CUDA_CHECK(cudaFreeHost(aHost));
}

/// Test large reductions (up to 1024^3 and larger).
template <typename Func>
void testLargeArgMax(const char *name, Func func, int N) {
    double *aHost;
    double *aDev;
    Pair *bDev;

    CUDA_CHECK(cudaMallocHost(&aHost, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&aDev, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bDev, 1 * sizeof(Pair)));

    for (int i = 0; i < N; ++i) {
        // aHost[i] = (N + 13241LL * i * i) % 432141;
        aHost[i] = 10 * i;
    }
    std::mt19937 gen;
    std::shuffle(aHost, aHost + N, gen);
    CUDA_CHECK(cudaMemcpy(aDev, aHost, N * sizeof(double), cudaMemcpyHostToDevice));

    func(aDev, bDev, N);

    int expectedIdx = std::max_element(aHost, aHost + N) - aHost;
    Pair expected{aHost[expectedIdx], expectedIdx};
    Pair received;
    CUDA_CHECK(cudaMemcpy(&received, bDev, 1 * sizeof(Pair), cudaMemcpyDeviceToHost));
    if (expected.idx != received.idx || expected.max != received.max) {
        printf("large %s incorrect result:  N=%d  expected=%d %f  received=%d %f\n",
               name, N, expected.idx, expected.max, received.idx, received.max);
        exit(1);
    }

    CUDA_CHECK(cudaFree(bDev));
    CUDA_CHECK(cudaFree(aDev));
    CUDA_CHECK(cudaFreeHost(aHost));
}
