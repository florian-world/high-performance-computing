#pragma once

__global__ void initializationKernel(
        short x0, short x1, short *x,
        float * t, int numTrajectories, int numIters);

__global__ void dimerizationKernel(
        int pass, const float *u,
        short *x, float *t, int *iters, char *isSampleDone,
        float endTime, int omega, int numIters, int numSamples);

__global__ void reduceIsDoneKernel(const char *isSampleDone, int *blocksDoneCount, int numSamples);


// TODO: Add __global__ function prototypes here.
