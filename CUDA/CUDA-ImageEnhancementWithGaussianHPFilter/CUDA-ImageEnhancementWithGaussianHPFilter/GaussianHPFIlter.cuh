#pragma once
#include <cuComplex.h>

using namespace std;

void fftShift2D(cuDoubleComplex** data, int width, int height);

__global__ void computeHighPassKernel(double* H, int width, int height, double cutoff_frequency);

__global__ void applyFilterKernel(cuDoubleComplex* F_shifted, double* H, cuDoubleComplex* G, int width, int height);

cuDoubleComplex** gaussianHighPassFilterCUDA(cuDoubleComplex** F_shifted, int width, int height, double cutoff_frequency);

cuDoubleComplex** unsharpMaskingFrequencyDomain(
    cuDoubleComplex** F_unshifted, // Frequency data in "normal" layout (DC at top-left)
    int width,
    int height,
    double cutoff_frequency,
    double alpha);

// Test function for Gaussian High-Pass Filter
bool testUnsharpMasking();

