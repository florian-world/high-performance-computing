#include <cuda_runtime.h>

__global__ void computeForcesKernel(int N, const double3 *p, double3 *f) {
    const size_t block_lb = blockIdx.x * blockDim.x;
    const size_t tidx = threadIdx.x;
    const size_t idx = block_lb + threadIdx.x;
    const size_t block_ub = block_lb + blockDim.x;
    if (idx >= N)
        return;

    // TODO: Copy the code from `nbody_c.cu` and utilize shared memory.
    extern __shared__ double3 s[];
    s[tidx] = p[idx];

    __syncthreads();

    double3 tmp{0.0, 0.0, 0.0};
    for (int i = 0; i < N; ++i) {
        double dx;
        double dy;
        double dz;

        if (block_lb <= i && i < block_ub) {
            dx = s[i-block_lb].x - s[tidx].x;
            dy = s[i-block_lb].y - s[tidx].y;
            dz = s[i-block_lb].z - s[tidx].z;
        } else {
            dx = p[i].x - s[tidx].x;
            dy = p[i].y - s[tidx].y;
            dz = p[i].z - s[tidx].z;
        }
        // Instead of skipping the i == idx case, add 1e-150 to avoid division
        // by zero. (dx * inv_r will be exactly 0.0)
        double inv_r = rsqrt(1e-150 + dx * dx + dy * dy + dz * dz);
        double inv_r_3 = inv_r*inv_r*inv_r;
        tmp.x += dx * inv_r_3;
        tmp.y += dy * inv_r_3;
        tmp.z += dz * inv_r_3;
    }
    f[idx] = tmp;
}

void computeForces(int N, const double3 *p, double3 *f) {
    constexpr int numThreads = 1024;
    int numBlocks = (N + numThreads - 1) / numThreads;

    // TODO: Set the required shared memory size.
    //       Don't bother with checking errors here.
    size_t shmSize = numThreads * sizeof(double)*3; // 3 doubles per thread  
    computeForcesKernel<<<numBlocks, numThreads, shmSize>>>(N, p, f);
}
