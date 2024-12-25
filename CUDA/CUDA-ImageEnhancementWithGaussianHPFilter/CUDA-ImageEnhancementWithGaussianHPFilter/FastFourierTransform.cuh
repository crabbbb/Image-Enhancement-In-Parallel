#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>

#define M_PI 3.14159265358979323846

__global__ void BitReverseRowKernel(
    cuDoubleComplex* d_data,
    int length,       // how many elements per row transform
    int batchCount,   // how many rows/batches
    int log2Len       // log2(length)
);

__global__ void CooleyTukeyRowKernel(
    cuDoubleComplex* d_data,
    int length,
    int batchCount,
    int halfSize,
    int sign // -1 => forward, +1 => inverse
);

__global__ void ScaleRowKernel(
    cuDoubleComplex* d_data,
    int length,
    int batchCount,
    double scale
);

__global__ void TransposeKernel(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_in,
    int width,
    int height
);

static void BatchFFT_1D(
    cuDoubleComplex* d_data,
    int length,
    int batches,
    bool forward
);

static void FFT2D_CUDA(
    cuDoubleComplex* d_data,
    int width,
    int height,
    bool forward
);

static void flattenHostArray(
    cuDoubleComplex** input2D,
    cuDoubleComplex* output1D,
    int width,
    int height
);

static void unflattenHostArray(
    const cuDoubleComplex* input1D,
    cuDoubleComplex** output2D,
    int width,
    int height
);

static void printComplex2D(
    cuDoubleComplex** arr,
    int w, int h,
    const char* msg,
    bool showImag = false
);

cuDoubleComplex** FFT2DParallel(cuDoubleComplex** inputImage, int width, int height);

cuDoubleComplex** IFFT2DParallel(cuDoubleComplex** freqData, int width, int height);


