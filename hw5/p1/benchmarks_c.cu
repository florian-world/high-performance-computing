#include "utils.h"
#include <numeric>
#include <omp.h>
#include <vector>
#include <cassert>

using ll = unsigned long long;

// Compute the sum of the Leibniz series. Each thread takes care of a subset of terms.
__global__ void leibnizKernel(ll K, double *partialSums) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    int numThreads = blockDim.x * gridDim.x;

    int iterPerThread = K / numThreads;

    // TODO: Compute the partial sum. Pick however you like which terms are computed by which thread.
    //       Avoid using std::pow for computing (-1)^k!

    ll k0 = idx*iterPerThread;
    ll k1 = k0 + iterPerThread;

    for (ll k = k0; k < k1; ++k) {
        if (k%2==0)
            sum += 1./(2*k+1);
        else
            sum -= 1./(2*k+1);
    }

    partialSums[idx] = sum;
}

/// Run the CUDA code for the given number of blocks and threads/block.
void runCUDA(ll K, int numBlocks, int threadsPerBlock) {
    int numThreads = numBlocks * threadsPerBlock;

    // Allocate the device and host buffers.

    double *partialSumsDev;
    double *partialSumsHost;

    // TODO: Allocate the temporary buffers for partial sums.
    CUDA_CHECK(cudaMalloc(&partialSumsDev, numThreads * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&partialSumsHost, numThreads * sizeof(double)));

    // TODO: Run the kernel and benchmark execution time.
    double dt = benchmark(5, [K, numBlocks, threadsPerBlock, partialSumsDev](){
        CUDA_LAUNCH(leibnizKernel, numBlocks, threadsPerBlock, K, partialSumsDev);
    });

    // just to make sure (already done in benchmark)
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemcpy(partialSumsHost, partialSumsDev, numThreads * sizeof(double), cudaMemcpyDeviceToHost));

    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < numThreads; ++i)
        sum += partialSumsHost[i];

    double pi = 4 * sum;
    printf("CUDA blocks=%5d  threads/block=%4d  iter/thread=%5lld  pi=%.12f  rel error=%.2g  Gterms/s=%.1f\n",
           numBlocks, threadsPerBlock, K / numThreads, pi, (pi - M_PI) / M_PI,
           1e-9 * K / dt);

    // TODO: Deallocate cuda buffers.
    cudaFree(partialSumsDev);
    cudaFreeHost(partialSumsHost);
}

/// Run the OpenMP variant of the code.
void runOpenMP(ll K, int numThreads) {
    double sum = 0.0;

    auto t0 = std::chrono::steady_clock::now();

    // TODO: Implement the Leibniz series summation with OpenMP.
    #pragma omp parallel for reduction(+:sum) num_threads(numThreads)
    for (ll k = 0; k < K; ++k) {
        if (k%2==0)
            sum += 1./(2*k+1);
        else
            sum -= 1./(2*k+1);
    }
    
    auto t1 = std::chrono::steady_clock::now();

    // TODO: Benchmark execution time.
    double dt = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-9;


    double pi = 4 * sum;
    printf("OpenMP threads=%d  pi=%.16g  rel error=%.2g  Gterms/s=%.1f\n",
           numThreads, pi, (pi - M_PI) / M_PI, 1e-9 * K / dt);
};


void subtask_c() {
    constexpr ll K = 2LL << 30;

    // TODO: Experiment with number of threads per block, and number of blocks
    // (i.e. number of iterations per thread).
    runCUDA(K, 8192, 1024);

    runOpenMP(K, 12);
}

int main() {
    subtask_c();
}
