// Do not edit!
#include "utils.h"
#include <chrono>
#include <random>

/// Depending on which target is compiling, different .o object file will be
/// linked.
void computeForces(int N, const double3 *pDev, double3 *fDev);

int main() {
    const int M = 60;         // Lattice size.
    const int N = M * M * M;  // Number of atoms.
    const double L = 100.0;   // Domain size.
    const int numSteps = 4;

    double3 *pDev;
    double3 *fDev;
    double3 *pHost;
    double3 *fHost;
    CUDA_CHECK(cudaMalloc(&pDev, N * sizeof(double3)));
    CUDA_CHECK(cudaMalloc(&fDev, N * sizeof(double3)));
    CUDA_CHECK(cudaMallocHost(&pHost, N * sizeof(double3)));
    CUDA_CHECK(cudaMallocHost(&fHost, N * sizeof(double3)));

    // Generate a lattice of points (+ noise).
    std::mt19937 gen{12345};
    const double h = L / M;
    std::uniform_real_distribution<double> distr(-0.1 * h, 0.1 * h);
    for (int iz = 0; iz < M; ++iz)
    for (int iy = 0; iy < M; ++iy)
    for (int ix = 0; ix < M; ++ix) {
        const int i = ix + M * (iy + M * iz);
        pHost[i] = double3{ix * h + distr(gen),
                           iy * h + distr(gen),
                           iz * h + distr(gen)};
    }

    // Upload initial state to the GPU.
    CUDA_CHECK(cudaMemcpy(pDev, pHost, N * sizeof(double3), cudaMemcpyHostToDevice));

    auto compute = [=]() {
        computeForces(N, pDev, fDev);
    };

    // Verification.
    compute();
    CUDA_CHECK(cudaMemcpy(fHost, fDev, N * sizeof(double3), cudaMemcpyDeviceToHost));
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    double ff = 0.0;
    for (int i = 0; i < N; ++i) {
        fx += fHost[i].x;
        fy += fHost[i].y;
        fz += fHost[i].z;
        ff += fHost[i].x * fHost[i].x + fHost[i].y * fHost[i].y + fHost[i].z * fHost[i].z;
    }
    printf("Statistics: <F>=%g %g %g  <F^2>=%g\n", fx, fy, fz, ff);

    // More warmup.
    for (int step = 0; step < 1; ++step)
        compute();

    // Benchmark.
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step < numSteps; ++step)
        compute();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::steady_clock::now();
    double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    double seconds_per_ts = 1e-9 * ns / numSteps;
    printf("Average execution time: %.1f ms\n", 1e3 * seconds_per_ts);

    CUDA_CHECK(cudaFreeHost(fHost));
    CUDA_CHECK(cudaFreeHost(pHost));
    CUDA_CHECK(cudaFree(fDev));
    CUDA_CHECK(cudaFree(pDev));
    return 0;
}
