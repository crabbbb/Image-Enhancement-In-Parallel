#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>
#include <cmath>
#include "device_launch_parameters.h"

#define M_PI 3.14159265358979323846

// --------------------------
// Bit-Reversal Helper Routines
// --------------------------
int bitReverseIndex(int x, int log2N) {
    int reversed = 0;
    for (int i = 0; i < log2N; i++) {
        reversed = (reversed << 1) | (x & 1);
        x >>= 1;
    }
    return reversed;
}

void bitReverseReorder(cuDoubleComplex* data, int N) {
    // Compute log2(N)
    int log2N = 0;
    while ((1 << log2N) < N) {
        log2N++;
    }

    // Swap each element with its bit-reversed index
    for (int i = 0; i < N; i++) {
        int r = bitReverseIndex(i, log2N);
        if (r > i) {
            cuDoubleComplex temp = data[i];
            data[i] = data[r];
            data[r] = temp;
        }
    }
}

// -------------------------------------------------------
// CUDA Kernel for one pass of the iterative Cooley-Tukey
//
// halfSize is the "sub-block" size for this pass; full
// butterfly size is 2*halfSize. For example, if halfSize=2,
// the butterfly size is 4. We do pairings inside each block
// of 4 elements.
// -------------------------------------------------------
__global__ void FFT1DKernel(cuDoubleComplex* data, int size, int halfSize)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalButterflies = size / 2; // total # of pairs in each pass

    if (tid < totalButterflies) {
        // For each thread, figure out which "block of 2*halfSize" we're in
        // and which index inside the "lower half" of that block
        int blockIndex = tid / halfSize;     // which block of size 2*halfSize
        int t = tid % halfSize;     // position within that block's lower half

        int butterflySize = 2 * halfSize;    // size of the local butterfly block

        // evenIndex and oddIndex
        int evenIndex = blockIndex * butterflySize + t;
        int oddIndex = evenIndex + halfSize;

        cuDoubleComplex even = data[evenIndex];
        cuDoubleComplex odd = data[oddIndex];

        // Twiddle factor = exp(-2*pi*i * t / (2*halfSize))
        double angle = -2.0 * M_PI * t / (double)butterflySize;
        cuDoubleComplex twiddle = make_cuDoubleComplex(cos(angle), sin(angle));

        cuDoubleComplex product = cuCmul(twiddle, odd);

        data[evenIndex] = cuCadd(even, product); // even + odd
        data[oddIndex] = cuCsub(even, product); // even - odd
    }
}

// --------------------------
// Utility to convert grayscale to complex
// --------------------------
cuDoubleComplex* convertToComplex(const uint8_t* grayscale, int size) {
    cuDoubleComplex* complexArray = new cuDoubleComplex[size];
    for (int i = 0; i < size; ++i) {
        complexArray[i] = make_cuDoubleComplex((double)grayscale[i], 0.0);
    }
    return complexArray;
}

// --------------------------
// Host Function for Parallelized 1D FFT
// --------------------------
void FFT1DParallel(cuDoubleComplex* h_input, cuDoubleComplex* h_output, int size) {
    // 1) Reorder (bit-reversal) on the host
    bitReverseReorder(h_input, size);

    // 2) Copy host data to device
    cuDoubleComplex* d_data = nullptr;
    cudaMalloc(&d_data, size * sizeof(cuDoubleComplex));
    cudaMemcpy(d_data, h_input, size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // 3) Define CUDA config
    //    We'll launch (size/2) threads in total. Each thread handles one "butterfly pair."
    int threadsPerBlock = 256;
    int totalButterflies = size / 2;
    int blocksPerGrid = (totalButterflies + threadsPerBlock - 1) / threadsPerBlock;

    // 4) Perform the iterative passes
    for (int halfSize = 1; halfSize < size; halfSize *= 2) {
        FFT1DKernel << <blocksPerGrid, threadsPerBlock >> > (d_data, size, halfSize);
        cudaDeviceSynchronize();
    }

    // 5) Copy result back
    cudaMemcpy(h_output, d_data, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // 6) Cleanup
    cudaFree(d_data);
}

// --------------------------
// 2D FFT Implementation
// --------------------------
cuDoubleComplex** FFT2DParallel(const uint8_t* grayscaleImage, int width, int height)
{
    // Allocate container for 2D result on the host
    cuDoubleComplex** fft_result = new cuDoubleComplex * [height];
    for (int i = 0; i < height; ++i) {
        fft_result[i] = new cuDoubleComplex[width];
    }

    // 1) Row-wise FFT
    for (int i = 0; i < height; ++i) {
        // Convert row i to complex
        cuDoubleComplex* row_data = convertToComplex(&grayscaleImage[i * width], width);

        // Parallel 1D FFT on that row
        FFT1DParallel(row_data, fft_result[i], width);

        delete[] row_data;
    }

    // 2) Column-wise FFT
    cuDoubleComplex* colData = new cuDoubleComplex[height];
    cuDoubleComplex* colFFT = new cuDoubleComplex[height];

    for (int j = 0; j < width; ++j) {
        // Gather the j-th column into colData[]
        for (int i = 0; i < height; ++i) {
            colData[i] = fft_result[i][j];
        }

        // Parallel 1D FFT on that column
        FFT1DParallel(colData, colFFT, height);

        // Scatter back into fft_result
        for (int i = 0; i < height; ++i) {
            fft_result[i][j] = colFFT[i];
        }
    }

    delete[] colData;
    delete[] colFFT;

    return fft_result;
}

// --------------------------
// Test Function for 2D FFT
// --------------------------
bool testFFT2DParallel() {
    const int width = 4;
    const int height = 4;

    uint8_t image[width * height] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    std::cout << "Input Grayscale Image:\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << (int)image[i * width + j] << " ";
        }
        std::cout << "\n";
    }

    // Perform 2D FFT
    cuDoubleComplex** fft_result = FFT2DParallel(image, width, height);

    // Print the result
    std::cout << "\n2D FFT Output (Complex Values):\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            double re = cuCreal(fft_result[i][j]);
            double im = cuCimag(fft_result[i][j]);
            std::cout << "(" << re << ", " << im << ") ";
        }
        std::cout << "\n";
    }

    // Cleanup
    for (int i = 0; i < height; ++i) {
        delete[] fft_result[i];
    }
    delete[] fft_result;

    return true;
}
