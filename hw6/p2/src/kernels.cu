#include "kernels.h"

/// Initialize reaction states.
__global__ void initializationKernel(
        short Sa, short Sb, short *x,
        float *t, int numSamples, int numIters)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numSamples)
        return;

    // Every sample starts with (Sa, Sb) at t=0.
    t[idx] = 0.f;
    x[0 * numIters * numSamples + idx] = Sa;
    x[1 * numIters * numSamples + idx] = Sb;
}


/// Reaction simulation. This kernel uses precomputed random uniform samples
/// (from 0 to 1) to compute up to `numIters` steps of the SSA algorithm. The
/// values of Sa and Sb are stored in `x`, the time values in `t`. Buffer
/// `iters` stores the number of performed iterations, and `isSampleDone`
/// whether or not the sample has reach the final state (t >= endTime).
__global__ void dimerizationKernel(
        int pass, const float *u,
        short *x, float *t, int *iters, char *isSampleDone,
        float endTime, int omega, int numIters, int numSamples)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int rngOffset = blockIdx.x * blockDim.x * 2 * numIters + threadIdx.x;

    if (idx >= numSamples)
        return;

    // Reaction rates.
    const float k1 = 1;
    const float k2 = 1;
    const float k3 = 0.2f / omega;
    const float k4 = 20.f * omega;

    // State variables.
    float time;
    float Sa, Sb;

    // Load state.
    const bool continuation = pass > 0 && !isSampleDone[idx];
    if (continuation) {
        Sa   = x[0 * numIters * numSamples + (numIters - 1) * numSamples + idx];
        Sb   = x[1 * numIters * numSamples + (numIters - 1) * numSamples + idx];
        time = t[(numIters - 1) * numSamples + idx];
    } else {
        Sa   = x[0 * numIters * numSamples + idx];
        Sb   = x[1 * numIters * numSamples + idx];
        time = t[idx];
    }

    // Simulation loop.
    int iter;
    for (iter = 0; time < endTime && iter < numIters && (pass == 0 || !isSampleDone[idx]); ++iter) {
        // Accumulated propensities.
        const float a1 = k1*Sa;
        const float a2 = a1 + k2*Sb;
        const float a3 = a2 + k3*Sa*Sb;
        const float a4 = a3 + k4;
        const float a0 = a4;

        time -= 1 / a0 * log(u[rngOffset]);
        rngOffset += blockDim.x;

        const float beta = a0 * u[rngOffset];
        rngOffset += blockDim.x;

        const int d1 = (beta < a1);
        const int d2 = (beta >= a1 && beta < a2);
        const int d3 = (beta >= a2 && beta < a3);
        const int d4 = (beta >= a3);

        Sa += -d1 + d3;
        Sb += -d2 - d3 + d4;

        t[iter * numSamples + idx] = time;
        x[0 * numIters * numSamples + iter * numSamples + idx] = Sa;
        x[1 * numIters * numSamples + iter * numSamples + idx] = Sb;
    }

    // Termination markers.
    iters[idx]        = iter;
    isSampleDone[idx] = time >= endTime || isSampleDone[idx];
}



__device__ int countWarp(char isSampleDone) {

    unsigned bitCounts = __ballot_sync(0xFFFFFFFF, isSampleDone);

    return __popc(bitCounts);
}

// from Q1
__device__ int sumWarp(int a) {
    int sum = a;
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 1);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, 16);
    return sum;
}


/// Store the sum of the subarray isSampleDone[1024*b : 1024*b+1023] in blocksDoneCount[b].
__global__ void reduceIsDoneKernel(const char *isSampleDone, int *blocksDoneCount, int numSamples) {
    // TODO: Implement the reduction that computes how many samples in a block have completed.
    //       isSampleDone[sampleIdx] = 1 if sample has finished, 0 if not.
    //       blocksDoneCount[blockIdx] = 0..threads-1 (the value to compute).
    //       Feel free to reuse the code from Q1.
    //
    // ...
    // ...
    // ...
    // ...
    // ...

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    char done = idx < numSamples ? isSampleDone[idx] : 0;

    int count = countWarp(done);

    __shared__ int sdata[32];

    if (threadIdx.x % 32 == 0)
        sdata[threadIdx.x / 32] = count;
    __syncthreads();

    if (threadIdx.x < 32)
        count = sumWarp(sdata[threadIdx.x]);

    if (threadIdx.x == 0)
        blocksDoneCount[blockIdx.x] = count;
}


// TODO: Implement the binning mechanism.
//       
//       Add function prototypes to src/kernels.h, such that ssa.cu can access them.
// ...
// ...
// ...
// ...
// ...
// ...
// ...
// ...
// ...
// ...
