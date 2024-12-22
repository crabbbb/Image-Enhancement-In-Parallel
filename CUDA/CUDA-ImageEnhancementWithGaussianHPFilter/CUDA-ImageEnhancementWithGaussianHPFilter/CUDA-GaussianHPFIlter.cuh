#pragma once
#include <cuComplex.h>

using namespace std;

__global__ void computeHighPassKernel(double* H, int width, int height, double cutoff_frequency);

__global__ void applyFilterKernel(cuDoubleComplex* F_shifted, double* H, cuDoubleComplex* G, int width, int height);

cuDoubleComplex** gaussianHighPassFilterCUDA(cuDoubleComplex** F_shifted, int width, int height, double cutoff_frequency);

// Test function for Gaussian High-Pass Filter
bool testGaussianHighPassFilterCUDA();

