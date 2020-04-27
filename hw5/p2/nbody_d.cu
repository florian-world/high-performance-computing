#include <cuda_runtime.h>

__global__ void computeForcesKernel(int N, const double3 *p, double3 *f) {
    // TODO: Copy the code from `nbody_c.cu` and utilize shared memory.
}

void computeForces(int N, const double3 *p, double3 *f) {
    constexpr int numThreads = 1024;
    int numBlocks = (N + numThreads - 1) / numThreads;

    // TODO: Set the required shared memory size.
    //       Don't bother with checking errors here.
    computeForcesKernel<<<numBlocks, numThreads>>>(N, p, f);
}
