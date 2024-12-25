#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>

#define M_PI 3.14159265358979323846

int bitReverseIndex(int x, int log2N);

void bitReverseReorder(cuDoubleComplex* data, int N);

__global__ void CooleyTukey1DKernel(cuDoubleComplex* data, int size, int halfSize, int sign);

__global__ void ScaleKernel(cuDoubleComplex* data, int size, double scale);

void Do1DFFT(cuDoubleComplex* h_input, cuDoubleComplex* h_output, int size, bool forward);

void FFT1DParallel(cuDoubleComplex* h_input, cuDoubleComplex* h_output, int size);

cuDoubleComplex** FFT2DParallel(cuDoubleComplex** inputImage, int width, int height);

void IFFT1DParallel(cuDoubleComplex* h_input, cuDoubleComplex* h_output, int size);

cuDoubleComplex** IFFT2DParallel(cuDoubleComplex** freqData, int width, int height);

void testFFTAndIFFT();

