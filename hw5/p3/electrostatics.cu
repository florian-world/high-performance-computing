#include "utils.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <sys/stat.h>  // For mkdir...

/// Compute phi^(k+1} from phi^k, rho and h. For convenience we send hh=h^2 and invhh=1/h^2.
/// Here, `rho`, `phik` and `phik1` are row-major matrices, i.e. rho_{iy, ix} = rho[iy * N + ix].
__global__ void jacobiStepKernel(int N, double hh, double invhh,
                                 const double *rho, const double *phik, double *phik1) {
    // TODO: Task 3b) Compute phik1[iy * N + ix]. Don't forget about 0 boundary conditions!
}

void jacobiStep(int N, double h, const double *rhoDev, const double *phikDev, double *phik1Dev) {
    /// TODO: Task 3b) Invoke the kernel jacobiSetKernel. Consider using dim3 as number
    /// of blocks and number of threads!
}


__global__ void computeAphiKernel(int N, double invhh, const double *phi, double *Aphi) {
    // TODO: Task 3d) Compute Aphi[iy * N + ix]. Don't forget about 0 boundary conditions!
}

void computeAphi(int N, double h, const double *xDev, double *AphiDev) {
    /// TODO: Task 3d) Invoke the kernel `computeAphiKernel`.
}

// Print L1 and L2 error. Do not edit!
void printL1L2(int iter, int N, const double *AphiHost, const double *rhoHost) {
    double L1 = 0.0;
    double L2 = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double error = std::fabs(AphiHost[i] - (-rhoHost[i]));
        L1 += error;
        L2 += error * error;
    }
    printf("%05d  Aphi - -rho  ==>  L1=%10.5g  L2=%10.5g\n", iter, L1, L2);
}

/*
 * Dump a vector x to a csv file for visualization (see usage below). Do not edit!
 *
 * Usage:
 *     dumpCSV(N, someHostVector, iterationNumber);
 * Note: This function is very slow compared to the kernels.
 *       Run on every 1000th time step.
 */
void dumpCSV(int N, const double *xHost, int iter) {
    char filename[64];
    sprintf(filename, "output/dump-%05d.csv", iter);
    mkdir("output", 0777);
    FILE *f = fopen(filename, "w");
    if (f == nullptr) {
        fprintf(stderr, "Error opening file \"%s\".", filename);
        exit(1);
    }

    for (int iy = 0; iy < N; ++iy)
        for (int ix = 0; ix < N; ++ix)
            fprintf(f, ix == N - 1 ? "%g\n" : "%g,", xHost[iy * N + ix]);

    fclose(f);
}

int main() {
    const int N = 400;
    const int numIterations = 50000;
    const double L = 1.0;
    const double h = L / N;

    double *rhoDev;
    double *phikDev;
    double *phik1Dev;
    double *rhoHost;

    // TODO: Task 3b) Allocate buffers of N^2 doubles.
    //                (You might need additional temporary buffers to complete all tasks.)
    CUDA_CHECK(cudaMalloc(&rhoDev, N*N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&phikDev, N*N*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&phik1Dev, N*N*sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&rhoHost, N*N*sizeof(double)));


    // RHS with three non-zero elements at (x, y)=(0.3, 0.1), (0.4, 0.1) and (0.5, 0.6).
    for (int i = 0; i < N * N; ++i)
        rhoHost[i] = 0.0;
    rhoHost[(1 * N / 10) * N + (3 * N / 10)] = 2.0;
    rhoHost[(1 * N / 10) * N + (4 * N / 10)] = 1.0;
    rhoHost[(6 * N / 10) * N + (5 * N / 10)] = -2.0;
    // TODO: Task 3b) Upload rhoHost to rhoDev.

    CUDA_CHECK(cudaMemcpy(rhoDev, rhoHost, N*N*sizeof(double), cudaMemcpyHostToDevice));

    // Initial guess x^(0)_i = 0.
    // TODO: Task 3b) Memset phikDev to 0.
    CUDA_CHECK(cudaMemset(phikDev, 0, N*N*sizeof(double)));

    // TODO: Task 3c) Run the jacobiStep numIterations times.
    // TODO: Task 3d) Call computeAphi, download the result and call printL1L2.
    //                Ensure that L1 and L2 drop over time.
    // TODO: (OPTIONAL) Download the vector phik1 (or phik) and call dumpCSV for visualization.
    jacobiStep(N, h, rhoDev, phikDev, phik1Dev);

    // TODO: Task 3a) Deallocate buffers.
    cudaFree(rhoDev);
    cudaFree(phikDev);
    cudaFree(phik1Dev);
    cudaFreeHost(rhoHost);
}
