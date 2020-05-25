#include "utils.h"
#include <cassert>
#include <limits>

struct Pair {
    double max;
    int idx;
};

/// Find the maximum value `a` among all warps and return {max value, index of
/// the max}. The result must be correct on at least the 0th thread of each warp.
__device__ Pair argMaxWarp(double a) {
    // TODO: 1.b) Compute the argmax of the given value.
    //            Return the maximum and the location of the maximum (0..31).
    Pair result;
    result.max = 0.0;
    result.idx = 0;

    // ...

    return result;
}


/// Returns the argmax of all values `a` within a block,
/// with the correct answer returned at least by the 0th thread of a block.
__device__ Pair argMaxBlock(double a) {
    // TODO: 1.c) Compute the argmax of the given value.
    //            Return the maximum and the location of the maximum (0..1023).
    // NOTE: For 1.c) implement either this or `sumBlock`!
    Pair result;
    result.max = 0.0;
    result.idx = 0;

    // ...

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

