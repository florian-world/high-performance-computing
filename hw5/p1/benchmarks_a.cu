// TODO: (OPTIONAL) Implement `benchmark` in the utils file.
#include "utils.h"
#include <omp.h>

// TODO: Task 1a.1) create an empty kernel `emptyKernel`.


/// Invoke `emptyKernel` with given number of blocks and threads/block and
/// report the execution time.
void invokeEmpty(bool synchronize, int numBlocks, int threadsPerBlock) {
    // TODO: Benchmark invocation of the `emptyKernel` code with given number
    // of blocks and threads/block.

    double dt = 0.0;  // Time per invocation in seconds.
    printf("synchronize=%d blocks=%5d  threads/block=%4d  iteration=%.1f us\n",
           (int)synchronize, numBlocks, threadsPerBlock, 1e6 * dt);
};

/// Run an empty parallel region with `numThreads` threads.
void emptyParallelRegion(int numThreads) {
    // TODO: Task 1a.4) Add an OpenMP parallel region with `numThreads` threads.
    {
        // With this command we prevent the compiler from optimizing away the
        // whole parallel region.
        __asm__ volatile("");
    }
}

int main() {
    // Note: You don't have to follow this skeleton.

    invokeEmpty(false, 1, 1);  // Task 1a) #1
    invokeEmpty(true, 1, 1);   // Task 1a) #2
    invokeEmpty(true, 1, 32);  // Task 1a) #3
    invokeEmpty(true, 1, 1024);
    invokeEmpty(true, 32, 1024);
    invokeEmpty(true, 1024, 32);
    invokeEmpty(true, 32768, 1);
    invokeEmpty(true, 32768, 32);
    invokeEmpty(true, 32768, 1024);


    static constexpr int numThreads = 12;
    // TODO: Task 1a.4) Benchmark `emptyParallelRegion`.

    double dt = 0.0;  // Time per invocation in seconds.
    printf("Empty OpenMP parallel region with %d threads --> %.1f us\n",
           numThreads, 1e6 * dt);
}
