#include <cuda_runtime.h>
#include <cuComplex.h>
#include <iostream>
#include <cmath>
#include "device_launch_parameters.h"
#include "CUDA-Utils.hpp"

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
// A single kernel that can do either forward or inverse
// butterfly pass based on 'sign'.
//
// sign = -1 => forward FFT (exp(-i*...))
// sign = +1 => inverse FFT (exp(+i*...))
//
// halfSize is the "sub-block" size for this pass; full
// butterfly size is 2*halfSize.
// -------------------------------------------------------
__global__ void CooleyTukey1DKernel(cuDoubleComplex* data, int size, int halfSize, int sign)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalButterflies = size / 2; // total # of pairs in this pass

    if (tid < totalButterflies) {
        // Identify which block of size 2*halfSize we are in
        int blockIndex = tid / halfSize;
        int t = tid % halfSize;
        int butterflySize = 2 * halfSize;

        // evenIndex, oddIndex
        int evenIndex = blockIndex * butterflySize + t;
        int oddIndex = evenIndex + halfSize;

        cuDoubleComplex even = data[evenIndex];
        cuDoubleComplex odd = data[oddIndex];

        // angle depends on sign (-1 => forward, +1 => inverse)
        double angle = sign * 2.0 * M_PI * t / (double)butterflySize;
        cuDoubleComplex twiddle = make_cuDoubleComplex(cos(angle), sin(angle));

        cuDoubleComplex product = cuCmul(twiddle, odd);
        data[evenIndex] = cuCadd(even, product); // even + product
        data[oddIndex] = cuCsub(even, product); // even - product
    }
}

// -------------------------------------------------------
// GPU kernel to scale the entire array by a constant factor
// -------------------------------------------------------
__global__ void ScaleKernel(cuDoubleComplex* data, int size, double scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = make_cuDoubleComplex(
            cuCreal(data[idx]) * scale,
            cuCimag(data[idx]) * scale
        );
    }
}

// -------------------------------------------------------
// Single routine for 1D FFT or IFFT, controlled by `forward`.
//
// forward = true  => forward FFT
// forward = false => inverse FFT
//
// The only differences: 
//   - sign in the exponential ( -1 vs +1 )
//   - final scale ( if inverse, divide by size )
// -------------------------------------------------------
void Do1DFFT(cuDoubleComplex* h_input, cuDoubleComplex* h_output, int size, bool forward)
{
    // 1) Bit-reversal reorder on the host
    bitReverseReorder(h_input, size);

    // 2) Copy to device
    cuDoubleComplex* d_data = nullptr;
    cudaMalloc(&d_data, size * sizeof(cuDoubleComplex));
    cudaMemcpy(d_data, h_input, size * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // 3) Kernel config
    int threadsPerBlock = 256;
    int totalButterflies = size / 2;
    int blocksPerGrid = (totalButterflies + threadsPerBlock - 1) / threadsPerBlock;

    // sign = -1 for forward FFT, +1 for inverse FFT
    int sign = (forward ? -1 : +1);

    // 4) Iterative passes (log2(size) passes)
    for (int halfSize = 1; halfSize < size; halfSize *= 2) {
        CooleyTukey1DKernel << <blocksPerGrid, threadsPerBlock >> > (d_data, size, halfSize, sign);
        cudaDeviceSynchronize();
    }

    // 5) If inverse, scale by (1/size) to complete the normalization
    if (!forward) {
        int totalThreads = size;
        int scaleBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
        ScaleKernel << <scaleBlocks, threadsPerBlock >> > (d_data, size, 1.0 / (double)size);
        cudaDeviceSynchronize();
    }

    // 6) Copy result back
    cudaMemcpy(h_output, d_data, size * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // 7) Cleanup
    cudaFree(d_data);
}

// -------------------------------------------------------
// Forward 1D FFT wrapper (for convenience)
// -------------------------------------------------------
void FFT1DParallel(cuDoubleComplex* h_input, cuDoubleComplex* h_output, int size)
{
    Do1DFFT(h_input, h_output, size, true /* forward */);
}

// -------------------------------------------------------
// Forward 2D FFT Implementation
// -------------------------------------------------------
// -------------------------------------------------------
// Forward 2D FFT Implementation (receives cuDoubleComplex**)
// -------------------------------------------------------
cuDoubleComplex** FFT2DParallel(cuDoubleComplex** inputImage, int width, int height)
{
    // Allocate container for the final 2D FFT result
    cuDoubleComplex** fft_result = new cuDoubleComplex * [height];
    for (int i = 0; i < height; ++i) {
        fft_result[i] = new cuDoubleComplex[width];
    }

    // 1) Row-wise transform
    for (int i = 0; i < height; ++i) {
        // Copy row i from inputImage to a temporary buffer
        cuDoubleComplex* row_data = new cuDoubleComplex[width];
        for (int c = 0; c < width; ++c) {
            row_data[c] = inputImage[i][c];
        }

        // Forward 1D FFT on this row
        FFT1DParallel(row_data, fft_result[i], width);

        delete[] row_data;
    }

    // 2) Column-wise transform
    cuDoubleComplex* colData = new cuDoubleComplex[height];
    cuDoubleComplex* colFFT = new cuDoubleComplex[height];

    for (int j = 0; j < width; ++j) {
        // Gather the j-th column from fft_result
        for (int i = 0; i < height; ++i) {
            colData[i] = fft_result[i][j];
        }

        // Forward 1D FFT on this column
        FFT1DParallel(colData, colFFT, height);

        // Scatter results back into fft_result
        for (int i = 0; i < height; ++i) {
            fft_result[i][j] = colFFT[i];
        }
    }

    delete[] colData;
    delete[] colFFT;

    return fft_result;
}

// -------------------------------------------------------
// Inverse 1D FFT wrapper (for convenience)
// -------------------------------------------------------
void IFFT1DParallel(cuDoubleComplex* h_input, cuDoubleComplex* h_output, int size)
{
    Do1DFFT(h_input, h_output, size, false /* inverse */);
}

// -------------------------------------------------------
// Inverse 2D FFT Implementation
//
// Assumes 'freqData' is the output of a forward 2D FFT,
// i.e. the frequency domain representation.
//
// We'll do row-wise inverse, then column-wise inverse.
// Each 1D inverse transforms divides by its dimension,
// so the final image is scaled by (1 / (width * height)) overall.
// -------------------------------------------------------
cuDoubleComplex** IFFT2DParallel(cuDoubleComplex** freqData, int width, int height)
{
    // Allocate container for 2D IFFT result
    cuDoubleComplex** spatial_result = new cuDoubleComplex * [height];
    for (int i = 0; i < height; ++i) {
        spatial_result[i] = new cuDoubleComplex[width];
    }

    // 1) Perform inverse 1D transform row-wise
    for (int i = 0; i < height; ++i) {
        // Copy row i
        cuDoubleComplex* rowCopy = new cuDoubleComplex[width];
        for (int c = 0; c < width; ++c) {
            rowCopy[c] = freqData[i][c];
        }

        IFFT1DParallel(rowCopy, spatial_result[i], width);
        delete[] rowCopy;
    }

    // 2) Perform inverse 1D transform column-wise
    cuDoubleComplex* colData = new cuDoubleComplex[height];
    cuDoubleComplex* colIFFT = new cuDoubleComplex[height];

    for (int j = 0; j < width; ++j) {
        // Gather j-th column
        for (int i = 0; i < height; ++i) {
            colData[i] = spatial_result[i][j];
        }

        IFFT1DParallel(colData, colIFFT, height);

        // Put results back
        for (int i = 0; i < height; ++i) {
            spatial_result[i][j] = colIFFT[i];
        }
    }

    delete[] colData;
    delete[] colIFFT;

    return spatial_result;
}

// -------------------------------------------------------
// Test function demonstrating forward & inverse 2D FFT
// -------------------------------------------------------
void testFFTAndIFFT() {
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

    cuDoubleComplex** complex_image = convertUint8ToCuComplex2D(image, width, height);

    // 1) Forward 2D FFT
    cuDoubleComplex** fft_result = FFT2DParallel(complex_image, width, height);

    std::cout << "\n2D FFT Output (Complex Values):\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            double re = cuCreal(fft_result[i][j]);
            double im = cuCimag(fft_result[i][j]);
            std::cout << "(" << re << ", " << im << ") ";
        }
        std::cout << "\n";
    }

    // 2) Inverse 2D FFT (reconstruct spatial image)
    cuDoubleComplex** ifft_result = IFFT2DParallel(fft_result, width, height);

    std::cout << "\nInverse 2D FFT (Reconstructed) [real parts]:\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            double re = cuCreal(ifft_result[i][j]);
            // Typically we'd expect near integer values (1..16), so let's print them
            std::cout << re << " ";
        }
        std::cout << "\n";
    }

    // Cleanup
    for (int i = 0; i < height; ++i) {
        delete[] fft_result[i];
        delete[] ifft_result[i];
    }
    delete[] fft_result;
    delete[] ifft_result;

}
