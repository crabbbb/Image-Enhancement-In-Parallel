#include <cuda_runtime.h>
#include <iostream>
#include <cuComplex.h>
#include <cmath>
#include "device_launch_parameters.h"
#include "Utils.hpp"

using namespace std;

// Shift frequency data so that the DC component is moved from (0,0) to (height/2, width/2).
void fftShift2D(cuDoubleComplex** data, int width, int height)
{
    int h2 = height / 2;
    int w2 = width / 2;

    // Swap top-left with bottom-right
    for (int u = 0; u < h2; ++u) {
        for (int v = 0; v < w2; ++v) {
            std::swap(data[u][v], data[u + h2][v + w2]);
        }
    }

    // Swap top-right with bottom-left
    for (int u = 0; u < h2; ++u) {
        for (int v = w2; v < width; ++v) {
            std::swap(data[u][v], data[u + h2][v - w2]);
        }
    }
}

// CUDA Kernel to compute Gaussian High-Pass Filter values
__global__ void computeHighPassKernel(double* H, int width, int height, double cutoff_frequency) {
    int u = blockIdx.y * blockDim.y + threadIdx.y;
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < height && v < width) {
        double D_uv = sqrt(pow(u - height / 2.0, 2) + pow(v - width / 2.0, 2));
        H[u * width + v] = 1.0 - exp(-pow(D_uv, 2) / (2 * pow(cutoff_frequency, 2)));
    }
}

// CUDA Kernel to apply the filter to the frequency-domain data
__global__ void applyFilterKernel(cuDoubleComplex* F_shifted, double* H, cuDoubleComplex* G, int width, int height) {
    int u = blockIdx.y * blockDim.y + threadIdx.y;
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    if (u < height && v < width) {
        int idx = u * width + v;
        cuDoubleComplex value = make_cuDoubleComplex(H[idx] * cuCreal(F_shifted[idx]), H[idx] * cuCimag(F_shifted[idx]));
        G[idx] = value;
    }
}

// Host function to perform the Gaussian High-Pass Filter
cuDoubleComplex** gaussianHighPassFilterCUDA(cuDoubleComplex** F_shifted, int width, int height, double cutoff_frequency) {
    // Allocate memory for the filter on the device
    double* d_H;
    cudaMalloc(&d_H, width * height * sizeof(double));

    // Allocate memory for the input and output frequency-domain data
    cuDoubleComplex* d_F_shifted;
    cuDoubleComplex* d_G;
    cudaMalloc(&d_F_shifted, width * height * sizeof(cuDoubleComplex));
    cudaMalloc(&d_G, width * height * sizeof(cuDoubleComplex));

    // Flatten the 2D F_shifted array for CUDA
    cuDoubleComplex* h_F_shifted = new cuDoubleComplex[width * height];
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            h_F_shifted[i * width + j] = F_shifted[i][j];
        }
    }
    cudaMemcpy(d_F_shifted, h_F_shifted, width * height * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Define CUDA grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Compute the filter values
    computeHighPassKernel <<< gridSize, blockSize >>> (d_H, width, height, cutoff_frequency);
    cudaDeviceSynchronize();

    // Apply the filter
    applyFilterKernel <<< gridSize, blockSize >>> (d_F_shifted, d_H, d_G, width, height);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cuDoubleComplex* h_G = new cuDoubleComplex[width * height];
    cudaMemcpy(h_G, d_G, width * height * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // Convert the flattened result back to a 2D array
    cuDoubleComplex** G = new cuDoubleComplex * [height];
    for (int i = 0; i < height; ++i) {
        G[i] = new cuDoubleComplex[width];
        for (int j = 0; j < width; ++j) {
            G[i][j] = h_G[i * width + j];
        }
    }

    // Cleanup
    delete[] h_F_shifted;
    delete[] h_G;
    cudaFree(d_H);
    cudaFree(d_F_shifted);
    cudaFree(d_G);

    return G;
}

// New function for unsharp masking
cuDoubleComplex** unsharpMaskingFrequencyDomain(
    cuDoubleComplex** F_unshifted, // Frequency data in "normal" layout (DC at top-left)
    int width,
    int height,
    double cutoff_frequency,
    double alpha)
{
    // Shift F so that DC is at the center (this is required for a radial Gaussian HP).
    fftShift2D(F_unshifted, width, height);

    // 1. Get high-pass filtered version of the input image in frequency domain
    cuDoubleComplex** F_HP = gaussianHighPassFilterCUDA(F_unshifted, width, height, cutoff_frequency);

    // 2. Create a new 2D array to store the unsharp masked result: 
    //    F_sharp(u,v) = F(u,v) + alpha * F_HP(u,v)
    cuDoubleComplex** F_sharp = allocate2DArray(height, width);

    // 3. Combine original + scaled high-pass
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            // Use cuCmul for complex multiplication and cuCadd for addition
            F_sharp[u][v] = cuCadd(F_unshifted[u][v], cuCmul(make_cuDoubleComplex(alpha, 0.0), F_HP[u][v]));
        }
    }

    // Shift the result back so DC is at(0, 0) again
    fftShift2D(F_sharp, width, height);

    // 4. Clean up the HPF array if you want
    cleanup2DArray(F_HP, height);

    // Return the frequency-domain data of the sharpened image
    return F_sharp;
}

// Test function for CUDA Gaussian High-Pass Filter
bool testUnsharpMasking() {
    // Image dimensions
    const int width = 4;
    const int height = 4;

    // Cutoff frequency
    double cutoff_frequency = 100.0;
    double alpha = 1.0;

    // Simulated frequency-domain data (4x4 matrix)
    cuDoubleComplex** F_shifted = allocate2DArray(height, width);

    int value = 1;
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            F_shifted[u][v] = make_cuDoubleComplex(value, value);
            ++value;
        }
    }

    // Expected output (manually calculated for this example)
    cuDoubleComplex** expected = allocate2DArray(height, width);
    cuDoubleComplex** F_HP_expected = allocate2DArray(height, width);

    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            double D_uv = sqrt(pow(u - height / 2.0, 2) + pow(v - width / 2.0, 2));
            double H_uv = 1.0 - exp(-pow(D_uv, 2) / (2 * pow(cutoff_frequency, 2)));

            // Compute F_HP_expected[u][v] = F_shifted[u][v] * H_uv
            F_HP_expected[u][v] = cuCmul(F_shifted[u][v], make_cuDoubleComplex(H_uv, 0.0));

            // Compute expected[u][v] = F_shifted[u][v] + alpha * F_HP_expected[u][v]
            expected[u][v] = cuCadd(F_shifted[u][v],
                cuCmul(make_cuDoubleComplex(alpha, 0.0), F_HP_expected[u][v]));
        }
    }

    // Call the unsharpMasking function which will apply the Gaussian filter to the original image
    cuDoubleComplex** G = unsharpMaskingFrequencyDomain(F_shifted, width, height, cutoff_frequency, alpha);

    // Compare the output with the expected result
    bool test_passed = true;
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            // Allow for small floating-point differences
            double real_diff = fabs(cuCreal(G[u][v]) - cuCreal(expected[u][v]));
            double imag_diff = fabs(cuCimag(G[u][v]) - cuCimag(expected[u][v]));

            if (real_diff > 1e-5 || imag_diff > 1e-5) {
                test_passed = false;
                std::cout << "Mismatch at (" << u << ", " << v << "): "
                    << "Expected (" << cuCreal(expected[u][v]) << ", " << cuCimag(expected[u][v])
                    << "), Got (" << cuCreal(G[u][v]) << ", " << cuCimag(G[u][v]) << ")" << std::endl;
            }
        }
    }

    // Cleanup dynamically allocated arrays
    for (int i = 0; i < height; ++i) {
        delete[] F_shifted[i];
        delete[] G[i];
    }
    delete[] F_shifted;
    delete[] G;

    return test_passed;
}

