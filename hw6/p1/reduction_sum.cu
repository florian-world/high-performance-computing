 #include "utils.h"
#include <cassert>
#include <algorithm>

/// Returns the sum of all values `a` within a warp,
/// with the correct answer returned only by the 0th thread of a warp.
__device__ double sumWarp(double a) {
    // TODO: 1.a) Compute sum of all values within a warp.
    //            Only the threads with threadIdx.x % warpSize == 0 have to
    //            return the correct result.
    //            (although this function operates only on a single warp, it
    //            will be called with many threads for testing)
    double sum = a;
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 16);
    return sum;
}

/// Returns the sum of all values `a` within a block,
/// with the correct answer returned only by the 0th thread of a block.
__device__ double sumBlock(double a) {
    // TODO: 1.c) Compute the sum of values `a` for all threads within a block.
    //            Only threadIdx.x == 0 has to return the correct result.
    // NOTE: For 1.c) implement either this or `argMaxBlock`!

    // we are sure that there are 1024 threads all with meaningful data
    double result = sumWarp(a);

    __shared__ double sdata[32];

    if (threadIdx.x % 32 == 0)
        sdata[threadIdx.x / 32] = result;
    __syncthreads();

    if (threadIdx.x < 32) {
        result = sumWarp(sdata[threadIdx.x]);
    }

    return result;
}

/// Compute the sum of all values aDev[0]..aDev[N-1] for N <= 1024^2 and store the result to bDev[0].
void sum1M(const double *aDev, double *bDev, int N) {
    assert(N <= 1024 * 1024);
    // TODO: 1.d) Implement either this or `argMax1M`.
    //            Avoid copying any data back to the host.
    //            Hint: The solution requires more CUDA operations than just
    //            calling a single kernel. Feel free to use whatever you find
    //            necessary.
}


#include "reduction_sum.h"


int main() {
    testSmallSum(sumWarpTestKernel, sumWarpCheck, 32, 3);
    testSmallSum(sumWarpTestKernel, sumWarpCheck, 32, 32);
    testSmallSum(sumWarpTestKernel, sumWarpCheck, 32, 320);
    testSmallSum(sumWarpTestKernel, sumWarpCheck, 32, 1023123);
    printf("sumWarp OK.\n");

    /*
    // OPTIONAL: 1a reduce-all. In case you want to try to implement it,
    //           implement a global function `__device__ double sumWarpAll(double x)`,
    //           and comment out sumWarpAll* functions in utils.h.
    testSmallSum(sumWarpAllTestKernel, sumWarpAllCheck, 1, 3);
    testSmallSum(sumWarpAllTestKernel, sumWarpAllCheck, 1, 32);
    testSmallSum(sumWarpAllTestKernel, sumWarpAllCheck, 1, 320);
    testSmallSum(sumWarpAllTestKernel, sumWarpAllCheck, 1, 1023123);
    printf("sumWarpAll OK.\n");
    */

    testSmallSum(sumBlockTestKernel, sumBlockCheck, 1024, 32);
    testSmallSum(sumBlockTestKernel, sumBlockCheck, 1024, 1024);
    testSmallSum(sumBlockTestKernel, sumBlockCheck, 1024, 12341);
    testSmallSum(sumBlockTestKernel, sumBlockCheck, 1024, 1012311);
    printf("sumBlock OK.\n");

    testLargeSum("sum1M", sum1M, 32);
    testLargeSum("sum1M", sum1M, 1024);
    testLargeSum("sum1M", sum1M, 12341);
    testLargeSum("sum1M", sum1M, 1012311);
    printf("sum1M OK.\n");
}
