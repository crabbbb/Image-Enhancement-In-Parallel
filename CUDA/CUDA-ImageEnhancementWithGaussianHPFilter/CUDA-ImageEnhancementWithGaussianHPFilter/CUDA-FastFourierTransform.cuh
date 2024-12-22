#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>

#define M_PI 3.14159265358979323846

int bitReverseIndex(int x, int log2N);

void bitReverseReorder(cuDoubleComplex* data, int N);

// CUDA Kernel for Cooley-Tukey FFT
__global__ void FFT1DKernel(cuDoubleComplex* data, int size, int step);

// Host Function for Parallelized 1D FFT
void FFT1DParallel(cuDoubleComplex* h_input, cuDoubleComplex* h_output, int size);

// 2D FFT Implementation
cuDoubleComplex** FFT2DParallel(const uint8_t* grayscaleImage, int width, int height);

// Test Function for 2D FFT
bool testFFT2DParallel();

cuDoubleComplex* convertToComplex(uint8_t* grayscale, int size);

