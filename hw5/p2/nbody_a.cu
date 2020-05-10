#include <cuda_runtime.h>

__global__ void computeForcesKernel(int N, const double3 *p, double3 * __restrict__ f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // TODO: Copy the code from `nbody_0.cu` and fix the redundant memory accesses.

    f[idx] = double3{0.0, 0.0, 0.0};
    for (int i = 0; i < N; ++i) {
        double dx = p[i].x - p[idx].x;
        double dy = p[i].y - p[idx].y;
        double dz = p[i].z - p[idx].z;
        // Instead of skipping the i == idx case, add 1e-150 to avoid division
        // by zero. (dx * inv_r will be exactly 0.0)
        double r = sqrt(1e-150 + dx * dx + dy * dy + dz * dz);
        double inv_r = 1 / r;
        f[idx].x += dx * inv_r * inv_r * inv_r;
        f[idx].y += dy * inv_r * inv_r * inv_r;
        f[idx].z += dz * inv_r * inv_r * inv_r;
    }
}

void computeForces(int N, const double3 *p, double3 *f) {
    constexpr int numThreads = 1024;
    int numBlocks = (N + numThreads - 1) / numThreads;
    computeForcesKernel<<<numBlocks, numThreads>>>(N, p, f);
}
