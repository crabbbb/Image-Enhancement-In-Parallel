#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include "device_launch_parameters.h"

#define M_PI 3.14159265358979323846

// CUDA Kernel for Cooley-Tukey FFT
__global__ void FFT1DKernel(float2* data, int size, int step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 2) {
        int evenIdx = tid * 2 * step;
        int oddIdx = evenIdx + step;
        int k = tid * step;

        float2 even = data[evenIdx];
        float2 odd = data[oddIdx];
        float angle = -2.0f * M_PI * k / size;
        float2 twiddle = make_float2(cosf(angle), sinf(angle));

        // Butterfly operation
        float2 temp = make_float2(twiddle.x * odd.x - twiddle.y * odd.y, twiddle.x * odd.y + twiddle.y * odd.x);
        data[evenIdx] = make_float2(even.x + temp.x, even.y + temp.y);
        data[oddIdx] = make_float2(even.x - temp.x, even.y - temp.y);
    }
}

// Host Function for Parallelized 1D FFT
void FFT1DParallel(float2* h_input, float2* h_output, int size) {
    // Allocate device memory
    float2* d_data;
    cudaMalloc(&d_data, size * sizeof(float2));
    cudaMemcpy(d_data, h_input, size * sizeof(float2), cudaMemcpyHostToDevice);

    // Define CUDA configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (size / 2 + threadsPerBlock - 1) / threadsPerBlock;

    // Perform FFT
    for (int step = 1; step < size; step *= 2) {
        FFT1DKernel << <blocksPerGrid, threadsPerBlock >> > (d_data, size, step);
        cudaDeviceSynchronize();
    }

    // Copy the result back to host
    cudaMemcpy(h_output, d_data, size * sizeof(float2), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_data);
}

// 2D FFT Implementation
float2** FFT2D(uint8_t* grayscaleImage, int width, int height) {
    // Allocate memory for 2D complex array
    float2** fft_result = new float2 * [height];
    for (int i = 0; i < height; ++i) {
        fft_result[i] = new float2[width];
    }

    // Convert rows to float2
    float2* row_complex = new float2[width];
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            row_complex[j] = make_float2(static_cast<float>(grayscaleImage[i * width + j]), 0.0f);
        }

        // Perform FFT on the row
        FFT1DParallel(row_complex, fft_result[i], width);
    }

    // Perform column-wise FFT
    float2* column = new float2[height];
    float2* column_fft_result = new float2[height];
    for (int j = 0; j < width; ++j) {
        for (int i = 0; i < height; ++i) {
            column[i] = fft_result[i][j];
        }

        // Perform FFT on the column
        FFT1DParallel(column, column_fft_result, height);

        // Store the result
        for (int i = 0; i < height; ++i) {
            fft_result[i][j] = column_fft_result[i];
        }
    }

    // Normalize the result
    float normFactor = 1.0f / (width * height);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            fft_result[i][j].x *= normFactor;
            fft_result[i][j].y *= normFactor;
        }
    }

    // Cleanup
    delete[] row_complex;
    delete[] column;
    delete[] column_fft_result;

    return fft_result;
}

// Test Function for 2D FFT
bool testFFT2D() {
    const int width = 4;
    const int height = 4;

    // Example grayscale image
    uint8_t image[width * height] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    std::cout << "Input Grayscale Image:\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << static_cast<int>(image[i * width + j]) << " ";
        }
        std::cout << "\n";
    }

    // Perform 2D FFT
    float2** fft_result = FFT2D(image, width, height);

    // Output the FFT result
    std::cout << "2D FFT Output (Complex Values):\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << "(" << fft_result[i][j].x << ", " << fft_result[i][j].y << ") ";
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
