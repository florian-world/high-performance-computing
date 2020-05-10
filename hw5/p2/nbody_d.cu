#include <cuda_runtime.h>

__global__ void computeForcesKernel(int N, const double3 *p, double3 *f) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    double3 myp = idx < N ? p[idx] : double3{0.0, 0.0, 0.0};

    extern __shared__ double3 s[];

    // TODO: Copy the code from `nbody_c.cu` and utilize shared memory.
    double3 tmp{0.0, 0.0, 0.0};
    for (int i = 0; i < N; i+= blockDim.x) {

        // read into shared memory
        if (i+threadIdx.x < N) s[threadIdx.x] = p[i+threadIdx.x];

        __syncthreads();

        // work in shared memory from now on
        for (int j = 0; j < min(blockDim.x, N-i); ++j) {
            double dx = s[j].x - myp.x;
            double dy = s[j].y - myp.y;
            double dz = s[j].z - myp.z;
            // Instead of skipping the i == idx case, add 1e-150 to avoid division
            // by zero. (dx * inv_r will be exactly 0.0)
            double inv_r = rsqrt(1e-150 + dx * dx + dy * dy + dz * dz);
            double inv_r_3 = inv_r*inv_r*inv_r;
            tmp.x += dx * inv_r_3;
            tmp.y += dy * inv_r_3;
            tmp.z += dz * inv_r_3;
        }

        __syncthreads();
    }

    if (idx < N) f[idx] = tmp;
}

void computeForces(int N, const double3 *p, double3 *f) {
    constexpr int numThreads = 1024;
    int numBlocks = (N + numThreads - 1) / numThreads;

    // TODO: Set the required shared memory size.
    //       Don't bother with checking errors here.
    size_t shmSize = numThreads * sizeof(double3); // 3 doubles per thread  
    computeForcesKernel<<<numBlocks, numThreads, shmSize>>>(N, p, f);
}
