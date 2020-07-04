#include "utils.h"
#include <cassert>
#include <limits>

struct Pair {
    double max;
    int idx;
};

__device__ Pair argMax(const Pair& a1, unsigned int delta) {
    // if (other > 0.0)
    //     printf("Other is bigger, %f, in %d \n", a1.max, (a1.idx + delta));
    
    double other = __shfl_down_sync(0xFFFFFFFF, a1.max, delta);
    int other_id = __shfl_down_sync(0xFFFFFFFF, a1.idx, delta);

    if (a1.max > other)
        return a1;
    Pair ret { other, other_id};
    return ret;
}

/// Find the maximum value `a` among all warps and return {max value, index of
/// the max}. The result must be correct on at least the 0th thread of each warp.
__device__ Pair argMaxWarp(double a) {
    // TODO: 1.b) Compute the argmax of the given value.
    //            Return the maximum and the location of the maximum (0..31).
    double other;
    bool comp;
    int mask;
    Pair result;
    result.max = a;
    result.idx = threadIdx.x % 32;

    result = argMax(result, 1);
    result = argMax(result, 2);
    result = argMax(result, 4);
    result = argMax(result, 8);
    result = argMax(result, 16);

//     other = __shfl_xor_sync(0xFFFFFFFF, result.max, 1);
//     comp = other > result.max;
//     if (comp) {
//         result.max = other;
//         ++result.idx;
//     }
//     if (threadIdx.x < 32)
//         printf("Round %d: Thread %3d: %d\n", 1, threadIdx.x, result.idx);
//     other = __shfl_xor_sync(0xFFFFFFFF, result.max, 2);
//     comp = other > result.max;
//     mask = __ballot_sync(0xFFFFFFFF, comp);
//     if (comp) {
//         result.max = other;
//         result.idx = __shfl_down_sync(mask, result.idx, 2) + 2;
//     }
//     if (threadIdx.x < 32)
//         printf("Round %d: Thread %3d: %d\n", 2, threadIdx.x, result.idx);
//     other = __shfl_xor_sync(0xFFFFFFFF, result.max, 4);
//     comp = other > result.max;
//     mask = __ballot_sync(0xFFFFFFFF, comp);
//     if (comp) {
//         result.max = other;
//         result.idx = __shfl_down_sync(mask, result.idx, 4) + 4;
//     }
//     if (threadIdx.x < 32)
//         printf("Round %d: Thread %3d: %d\n", 4, threadIdx.x, result.idx);
//     other = __shfl_xor_sync(0xFFFFFFFF, result.max, 8);
//     comp = other > result.max;
//     mask = __ballot_sync(0xFFFFFFFF, comp);
//     if (comp) 
// {        result.max = other;
//         result.idx = __shfl_down_sync(mask, result.idx, 8) + 8;
//     }
//     if (threadIdx.x < 32)
//         printf("Round %d: Thread %3d: %d\n", 8, threadIdx.x, result.idx);
//     other = __shfl_xor_sync(0xFFFFFFFF, result.max, 16);
//     comp = other > result.max;
//     mask = __ballot_sync(0xFFFFFFFF, comp);
//     if (comp) {
//         result.max = other;
//         int shdown = __shfl_down_sync(mask, result.idx, 16);
//         if (threadIdx.x == 0)
//             printf("Thread 0 --> Comparison result %d, shuffle down: %d \n", comp, shdown);
//         result.idx = shdown + 16;
//     }
//     if (threadIdx.x < 32)
//         printf("Round %d: Thread %3d: %d\n", 16, threadIdx.x, result.idx);
    return result;
}


/// Returns the argmax of all values `a` within a block,
/// with the correct answer returned at least by the 0th thread of a block.
__device__ Pair argMaxBlock(double a) {
    // TODO: 1.c) Compute the argmax of the given value.
    //            Return the maximum and the location of the maximum (0..1023).
    // NOTE: For 1.c) implement either this or `sumBlock`!
    Pair result, blockResult;
    result.max = 0.0;
    result.idx = 0;

    // we are sure that there are 1024 threads all with meaningful data
    result = argMaxWarp(a);

    __shared__ double sdata[32];
    __shared__ int sdataidx[32];

    if (threadIdx.x % 32 == 0)
        sdata[threadIdx.x / 32] = result.max;
        sdataidx[threadIdx.x / 32] = result.idx;
    __syncthreads();

    if (threadIdx.x < 32) {
        blockResult = argMaxWarp(sdata[threadIdx.x]);
    }

    if (threadIdx.x == 0) {
        result.max = blockResult.max;
        result.idx = blockResult.idx*32 + sdataidx[blockResult.idx];
    }

    return result;
}


void argMax1M(const double *aDev, Pair *bDev, int N) {
    assert(N <= 1024 * 1024);
    // TODO: 1.d) Implement either this or `sum1M`.
    //            Avoid copying any data back to the host.
    //            Hint: The solution requires more CUDA operations than just
    //            calling a single kernel. Feel free to use whatever you find
    //            necessary.
}

#include "reduction_argmax.h"

int main() {

    printf("Some test here: %d, %d, %d, %d, %d\n", (1U<<1) - 1, (1U<<2) - 1, (1U<<3) - 1, (1U<<4) - 1, (1U<<5) - 1);

    testSmallArgMax(argMaxWarpTestKernel, argMaxWarpCheck, 32, 3);
    testSmallArgMax(argMaxWarpTestKernel, argMaxWarpCheck, 32, 32);
    testSmallArgMax(argMaxWarpTestKernel, argMaxWarpCheck, 32, 320);
    testSmallArgMax(argMaxWarpTestKernel, argMaxWarpCheck, 32, 1023123);
    printf("argMaxWarp OK.\n");

    testSmallArgMax(argMaxBlockTestKernel, argMaxBlockCheck, 1024, 32);
    testSmallArgMax(argMaxBlockTestKernel, argMaxBlockCheck, 1024, 1024);
    testSmallArgMax(argMaxBlockTestKernel, argMaxBlockCheck, 1024, 12341);
    testSmallArgMax(argMaxBlockTestKernel, argMaxBlockCheck, 1024, 1012311);
    printf("argMaxBlock OK.\n");

    testLargeArgMax("argMax1M", argMax1M, 32);
    testLargeArgMax("argMax1M", argMax1M, 1024);
    testLargeArgMax("argMax1M", argMax1M, 12341);
    testLargeArgMax("argMax1M", argMax1M, 1012311);
    printf("argMax1M OK.\n");
}

