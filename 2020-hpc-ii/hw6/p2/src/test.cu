#include "utils.h"
#include "kernels.h"
#include <random>

static std::mt19937 gen;

static void _testReduceKernel(int N, float probability) {
    int threads = 1024;
    int blocks = (N + threads - 1) / threads;

    char *isSampleDoneDev;
    int *perBlockDoneDev;
    char *isSampleDoneHost;
    int *perBlockDoneHost;
    CUDA_CHECK(cudaMalloc(&isSampleDoneDev, N * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&perBlockDoneDev, blocks * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&isSampleDoneHost, N * sizeof(char)));
    CUDA_CHECK(cudaMallocHost(&perBlockDoneHost, blocks * sizeof(int)));

    std::bernoulli_distribution distr(probability);
    for (int i = 0; i < N; ++i)
        isSampleDoneHost[i] = distr(gen);

    CUDA_CHECK(cudaMemcpy(isSampleDoneDev, isSampleDoneHost, N, cudaMemcpyHostToDevice));
    reduceIsDoneKernel<<<blocks, threads>>>(isSampleDoneDev, perBlockDoneDev, N);
    CUDA_CHECK(cudaMemcpy(perBlockDoneHost, perBlockDoneDev, blocks * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < blocks; ++i) {
        int expected = 0;
        for (int j = i * threads; j < std::min((i + 1) * threads, N); ++j)
            expected += (bool)isSampleDoneHost[j];
        int received = perBlockDoneHost[i];
        if (expected != received) {
            printf("    Test failed: blockDone[%d]  expected=%d  received=%d\n",
                   i, expected, received);
            exit(1);
        }
    }

    CUDA_CHECK(cudaFreeHost(isSampleDoneHost));
    CUDA_CHECK(cudaFree(isSampleDoneDev));
}

void testReduceIsDoneKernel() {
    printf("Testing blocksDoneKernel...\n");
    for (int i = 0; i < 5; ++i) {
        _testReduceKernel(13, 0.1);
        _testReduceKernel(13, 0.6);
        _testReduceKernel(32, 0.3);
        _testReduceKernel(256, 0);
        _testReduceKernel(1239811, 0.001);
        _testReduceKernel(1239811, 0.2);
        _testReduceKernel(1239811, 0.999);
    }
    printf("    Passed.\n");
}
