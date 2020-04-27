#include "utils.h"
#include <numeric>
#include <omp.h>
#include <vector>

using ll = long long;

// Compute the sum of the Leibniz series. Each thread takes care of a subset of terms.
__global__ void leibnizKernel(ll K, double *partialSums) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double sum = 0.0;

    // TODO: Compute the partial sum. Pick however you like which terms are computed by which thread.
    //       Avoid using std::pow for computing (-1)^k!

    partialSums[idx] = sum;
}

/// Run the CUDA code for the given number of blocks and threads/block.
void runCUDA(ll K, int numBlocks, int threadsPerBlock) {
    int numThreads = numBlocks * threadsPerBlock;

    // Allocate the device and host buffers.

    double *partialSumsDev;
    double *partialSumsHost;

    // TODO: Allocate the temporary buffers for partial sums.


    // TODO: Run the kernel and benchmark execution time.
    double dt = 0.0;


    // TODO: Copy the sumsDev to host and accumulate, and sum them up.
    double sum = 0.0;


    double pi = 4 * sum;
    printf("CUDA blocks=%5d  threads/block=%4d  iter/thread=%5lld  pi=%.12f  rel error=%.2g  Gterms/s=%.1f\n",
           numBlocks, threadsPerBlock, K / numThreads, pi, (pi - M_PI) / M_PI,
           1e-9 * K / dt);

    // TODO: Deallocate cuda buffers.
}

/// Run the OpenMP variant of the code.
void runOpenMP(ll K, int numThreads) {
    double sum = 0.0;

    // TODO: Implement the Leibniz series summation with OpenMP.


    // TODO: Benchmark execution time.
    double dt = 0.0;


    double pi = 4 * sum;
    printf("OpenMP threads=%d  pi=%.16g  rel error=%.2g  Gterms/s=%.1f\n",
           numThreads, pi, (pi - M_PI) / M_PI, 1e-9 * K / dt);
};


void subtask_c() {
    constexpr ll K = 2LL << 30;

    // TODO: Experiment with number of threads per block, and number of blocks
    // (i.e. number of iterations per thread).
    runCUDA(K, 512, 512);

    runOpenMP(K, 12);
}

int main() {
    subtask_c();
}
