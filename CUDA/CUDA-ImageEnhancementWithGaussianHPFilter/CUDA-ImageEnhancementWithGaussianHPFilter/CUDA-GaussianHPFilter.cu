#include <cuda_runtime.h>
#include <iostream>
#include <cuComplex.h>
#include <cmath>
#include "device_launch_parameters.h"

using namespace std;

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

// Test function for CUDA Gaussian High-Pass Filter
bool testGaussianHighPassFilterCUDA() {
    // Image dimensions
    const int width = 4;
    const int height = 4;

    // Cutoff frequency
    double cutoff_frequency = 2.0;

    // Simulated frequency-domain data (4x4 matrix)
    cuDoubleComplex** F_shifted = new cuDoubleComplex * [height];
    for (int i = 0; i < height; ++i) {
        F_shifted[i] = new cuDoubleComplex[width];
    }

    int value = 1;
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            F_shifted[u][v] = make_cuDoubleComplex(value, value);
            ++value;
        }
    }

    // Call the CUDA Gaussian High-Pass Filter
    cuDoubleComplex** G = gaussianHighPassFilterCUDA(F_shifted, width, height, cutoff_frequency);

    // Expected output (manually calculated for this example)
    bool test_passed = true;
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            double D_uv = sqrt(pow(u - height / 2.0, 2) + pow(v - width / 2.0, 2));
            double H_uv = 1.0 - exp(-pow(D_uv, 2) / (2 * pow(cutoff_frequency, 2)));
            cuDoubleComplex expected = make_cuDoubleComplex(
                cuCreal(F_shifted[u][v]) * H_uv,
                cuCimag(F_shifted[u][v]) * H_uv
            );

            // Allow for small floating-point differences
            if (abs(cuCreal(G[u][v]) - cuCreal(expected)) > 1e-6 ||
                abs(cuCimag(G[u][v]) - cuCimag(expected)) > 1e-6) {
                test_passed = false;
                cout << "Mismatch at (" << u << ", " << v << "): "
                    << "Expected (" << cuCreal(expected) << ", " << cuCimag(expected)
                    << "), Got (" << cuCreal(G[u][v]) << ", " << cuCimag(G[u][v]) << ")" << endl;
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

