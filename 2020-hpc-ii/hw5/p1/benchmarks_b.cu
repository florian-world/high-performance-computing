#include "utils.h"
#include <algorithm>
#include <random>

/// Buffer sizes we consider. The numbers are odd such that p[i]=(2*i)%K are all different.
static constexpr int kBufferSizes[] = {
    17, 65, 251, 1001, 2001, 5001,
    10'001, 25'001, 50'001, 100'001, 250'001, 500'001, 1'000'001,
    5'000'001, 20'000'001, 50'000'001,
};

__global__ void copyKernelABP(int K, double *a, const int* p, const double*b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        a[idx] = b[p[idx]];
    }
}

__global__ void copyKernelAPB(int K, double *a, const int* p, const double*b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        a[p[idx]] = b[idx];
    }
}

__global__ void addKernelAAB(int K, double *a, const double*b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        a[idx] = a[idx] + b[idx];
    }
}

__global__ void addKernelAAB100(int K, double *a, const double*b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K) {
        for (int i = 0; i < 100; ++i)
            a[idx] = a[idx] + b[idx];
    }
}




void subtask_b() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    const int threadsPerBlock = prop.maxThreadsPerBlock;

    printf("Using device with %d threads per block\n", threadsPerBlock);

    int maxK = kBufferSizes[sizeof(kBufferSizes) / sizeof(kBufferSizes[0]) - 1];

    /// Pick a N with respect to K such that total running time is more or less uniform.
    auto pickN = [](int K) {
        return 100'000 / (int)std::sqrt(K) + 5;  // Some heuristics.
    };

    double *aDev;
    double *bDev;
    int *pDev;
    double *aHost;
    int *pHost;

    // TODO: Allocate the buffers. Immediately allocate large enough buffers to handle the largest case (maxK).
    // Wrap all cuda APIs with the CUDA_CHECK macro, which will report if the API failed to execute.
    // For example,
    //      CUDA_CHECK(cudaMalloc(...));
    // CUDA_CHECK(cudaCmd) check whether `cudaCmd` completed successfully.
    CUDA_CHECK(cudaMalloc(&aDev, maxK * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bDev, maxK * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&pDev, maxK * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&aHost, maxK * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&pHost, maxK * sizeof(int)));

    // Set aDev, bDev and aHost to 0.0 (not really that important).
    CUDA_CHECK(cudaMemset(aDev, 0, maxK * sizeof(double)));
    CUDA_CHECK(cudaMemset(bDev, 0, maxK * sizeof(double)));
    memset(aHost, 0, maxK * sizeof(double));

    // Task 1b.1)
    for (int K : kBufferSizes) {
        // TODO: Measure the execution time of synchronously uploading K doubles from the host to the device. Report GB/s

        double dt = benchmark(pickN(K), [aDev, aHost, K](){
            CUDA_CHECK(cudaMemcpy(aDev, aHost, K * sizeof(double), cudaMemcpyHostToDevice));
        });

        double gbps = K*sizeof(double) / dt / 1e9; // Gigabytes per second here;
        // printf("upload K=%8d, averged over %8d runs --> %5.2f GB/s\n", K, pickN(K), gbps);
        printf("upload K=%8d --> %5.2f GB/s\n", K, gbps);
    }


    // Task 1b.2)
    /// Benchmark copying for a given access pattern (permutation).
    auto benchmarkPermutedCopy = [=](const char *description, auto permutationFunc) {
        for (int K : kBufferSizes) {
            // Compute the permutation p[i].
            permutationFunc(K);

            /// TODO: Copy pHost to pDev. Don't forget CUDA_CHECK.
            CUDA_CHECK(cudaMemcpy(pDev, pHost, K * sizeof(int), cudaMemcpyHostToDevice));

            /// TODO: Benchmark the a_i = b_{p_i} kernel.
            double dtABP = benchmark(pickN(K), [threadsPerBlock, K, aDev, bDev, pDev]() {
                CUDA_LAUNCH(copyKernelABP, (K+threadsPerBlock-1) / threadsPerBlock, threadsPerBlock, K, aDev, pDev, bDev);
            });

            /// TODO: (OPTIONAL) Benchmark the a_{p_i} = b_i kernel;
            double dtAPB = benchmark(pickN(K), [threadsPerBlock, K, aDev, bDev, pDev]() {
                CUDA_LAUNCH(copyKernelAPB, (K+threadsPerBlock-1) / threadsPerBlock, threadsPerBlock, K, aDev, pDev, bDev);
            });

            // Report how many bytes per second was written.
            printf("Case %s  -->  K=%8d  [a=b_p] %6.2f GB/s  [a_p=b] %6.2f GB/s written\n",
                   description, K,
                   1e-9 * K * sizeof(double) / dtABP,
                   1e-9 * K * sizeof(double) / dtAPB);
        }
    };

    // The patterns are already implemented, do not modify!
    std::mt19937 gen;
    benchmarkPermutedCopy("p[i]=i", [pHost](int K) {
        for (int i = 0; i < K; ++i)
            pHost[i] = i;
    });
    benchmarkPermutedCopy("p[i]=(2*i)%K", [pHost](int K) {
        for (int i = 0; i < K; ++i)
            pHost[i] = (2 * i) % K;
    });
    benchmarkPermutedCopy("p[i]=(4*i)%K", [pHost](int K) {
        for (int i = 0; i < K; ++i)
            pHost[i] = (4 * i) % K;
    });
    benchmarkPermutedCopy("p[i]=i, 32-shuffled", [pHost, &gen](int K) {
        for (int i = 0; i < K; ++i)
            pHost[i] = i;
        for (int i = 0; i < K; i += 32)
            std::shuffle(pHost + i, pHost + std::min(i + 32, K), gen);
    });
    benchmarkPermutedCopy("fully shuffled", [pHost, &gen](int K) {
        for (int i = 0; i < K; ++i)
            pHost[i] = i;
        std::shuffle(pHost, pHost + K, gen);
    });


    // Task 1b.3) and 1b.4)
    for (int K : kBufferSizes) {
        // TODO: Benchmark a_i += b_i kernel.
        double dt1 = benchmark(pickN(K), [threadsPerBlock, K, aDev, bDev]() {
            CUDA_LAUNCH(addKernelAAB, (K+threadsPerBlock-1) / threadsPerBlock, threadsPerBlock, K, aDev, bDev);
        });;

        // TODO: Benchmark the kernel that repeats a_i += b_i 100x times.
        double dt100 = benchmark(pickN(K), [threadsPerBlock, K, aDev, bDev]() {
            CUDA_LAUNCH(addKernelAAB100, (K+threadsPerBlock-1) / threadsPerBlock, threadsPerBlock, K, aDev, bDev);
        });;

        // printf("dts = %f, %f\n", dt1, dt100);

        double gflops1 = K*1E-9/dt1;
        double gflops100 = K*100*1E-9/dt100;
        printf("a+b  -->  K=%8d  1x -> %4.1f GFLOP/s  100x -> %5.1f GFLOP/s\n", K, gflops1, gflops100);
    }


    // TODO: Free all host and all device buffers.

    cudaFree(aDev);
    cudaFree(bDev);
    cudaFree(pDev);
    cudaFreeHost(aHost);
    cudaFreeHost(pHost);
}

int main() {
    subtask_b();
}
