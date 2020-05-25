#include "utils.h"
#include <cuda_profiler_api.h>
#include <algorithm>
#include <vector>

/// Memory-bound dummy kernel. Do not edit.
__global__ void fastKernel(const double *a, double *b, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M)
        return;

    b[idx] = 10.0 * a[idx];
}

/// Compute-bound dummy kernel. Do not edit.
__global__ void slowKernel(const double *a, double *b, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M)
        return;

    double x = a[idx];
    for (int i = 0; i < 10000; ++i)
        x *= 1.01;

    b[idx] = (x != 0.1231 ? 10.0 : -1.0) * a[idx];
}

/// Check whether `bHost` contains the correct result. Do not edit.
void checkResults(const double *bHost, int N) {
    for (int i = 0; i < N; ++i) {
        if (bHost[i] != 100.0 * i) {
            printf("Incorrect value for i=%d:  value before kernel=%.1f  "
                   "expected after=%.1f  now=%.1f\n",
                   i, 10.0 * i, 100. * i, bHost[i]);
            exit(1);
        }
    }
}




/// Asynchronously, and in chunks, copy the array to the device, execute the
/// kernel and copy the result back.
template <typename Kernel>
void runAsync(const char *kernelName, Kernel kernel, int N, int chunkSize, int numStreams) {
    double *aHost;
    double *bHost;

    CUDA_CHECK(cudaMallocHost(&aHost, N * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&bHost, N * sizeof(double)));
    for (int i = 0; i < N; ++i)
        aHost[i] = 10.0 * i;

    // TODO 3.a) Allocate chunks and create streams.


    // Instead of benchmark() we use a simplified measure() which invokes the
    // function only once (to get a cleaner profiling information).
    double dt = measure([&]() {
        // TODO 3.b) (1) Upload `a`, (2) launch the kernel, and (3) download
        //           `b` in chunks of size `chunkSize`.
        //           Use streams in a cyclic fashion.
        //
        //           Note: you can use CUDA_CHECK and CUDA_LAUNCH_EX from
        //           utils.h for error checking.

        // TODO 3.b) Synchronize the streams.
    });

    checkResults(bHost, N);

    printf("async %s  N=%9d  chunkSize=%9d  numStreams=%d  time=%fs\n",
           kernelName, N, chunkSize, numStreams, dt);

    // TODO: 3.a) Deallocate chunks and destroy streams.

    CUDA_CHECK(cudaFreeHost(bHost));
    CUDA_CHECK(cudaFreeHost(aHost));
}





/// Synchronously copy the whole array to the device, execute the kernel and
/// copy the result back. Do not edit.
template <typename Kernel>
void runSync(const char *kernelName, Kernel kernel, int N) {
    double *aHost;
    double *bHost;
    double *aDev;
    double *bDev;

    CUDA_CHECK(cudaMallocHost(&aHost, N * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&bHost, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&aDev, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bDev, N * sizeof(double)));
    for (int i = 0; i < N; ++i)
        aHost[i] = 10.0 * i;

    // Host -> device.
    double dt1 = measure([&]() {
        CUDA_CHECK(cudaMemcpy(aDev, aHost, N * sizeof(double), cudaMemcpyHostToDevice));
    });
    // Kernel.
    double dt2 = measure([&]() {
        // We cannot execute more than maxBlocks blocks, so we split the work
        // into multiple launches. That's another reason for using chunks.
        int threads = 1024;
        int maxBlocks = 65'536;
        int blocks = (N + threads - 1) / threads;
        for (int i = 0; i < blocks; i += maxBlocks) {
            CUDA_LAUNCH(kernel, std::min(maxBlocks, blocks - i), threads,
                        aDev + i * threads, bDev + i * threads, N);
        }
    });
    // Device -> host.
    double dt3 = measure([&]() {
        CUDA_CHECK(cudaMemcpy(bHost, bDev, N * sizeof(double), cudaMemcpyDeviceToHost));
    });

    checkResults(bHost, N);

    printf("sync  %s  N=%9d  upload=%fs  kernel=%fs  download=%fs  total=%fs\n",
           kernelName, N, dt1, dt2, dt3, dt1 + dt2 + dt3);

    CUDA_CHECK(cudaFree(bDev));
    CUDA_CHECK(cudaFree(aDev));
    CUDA_CHECK(cudaFreeHost(bHost));
    CUDA_CHECK(cudaFreeHost(aHost));
}

/// Selection of runs to use for profiling.
void profile() {
    runSync("fastKernel", fastKernel, 100'000'000);
    runAsync("fastKernel", fastKernel, 100'000'000, 10'000'000, 4);
    runSync("slowKernel", slowKernel, 100'000'000);
    runAsync("slowKernel", slowKernel, 100'000'000, 10'000'000, 4);
    cudaProfilerStop();
}

/// Selection of runs to use for benchmarking.
void runBenchmarks() {
    runSync("fastKernel", fastKernel, 1'000'000);
    runSync("fastKernel", fastKernel, 100'000'000);
    runAsync("fastKernel", fastKernel, 100'000'000, 100'000'000, 1);
    runAsync("fastKernel", fastKernel, 100'000'000, 10'000'000, 4);
    runAsync("fastKernel", fastKernel, 100'000'000, 10'000'000, 8);
    runAsync("fastKernel", fastKernel, 100'000'000, 1'000'000, 4);
    runAsync("fastKernel", fastKernel, 100'000'000, 1'000'000, 8);
    printf("\n");

    runSync("slowKernel", slowKernel, 1'000'000);
    runSync("slowKernel", slowKernel, 100'000'000);
    runAsync("slowKernel", slowKernel, 100'000'000, 100'000'000, 1);
    runAsync("slowKernel", slowKernel, 100'000'000, 10'000'000, 4);
    runAsync("slowKernel", slowKernel, 100'000'000, 10'000'000, 8);
    runAsync("slowKernel", slowKernel, 100'000'000, 1'000'000, 4);
    runAsync("slowKernel", slowKernel, 100'000'000, 1'000'000, 8);
}

int main() {
    // TODO: 3.c.) Enable `profile` and disable `runBenchmarks` to get a
    //             cleaner profiling information.

    // profile();
    runBenchmarks();
}
