// CUDA-ImageEnhancementWithGaussianHPFilter.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "CUDA-GaussianHPFIlter.cuh"
#include "CUDA-FastFourierTransform.cuh"
#include "CUDA-Utils.hpp"

using namespace std;

void testFFT2DToGaussianFilterToIFFT2D() {
    // Image dimensions
    const int width = 4;
    const int height = 4;

    // Simulated grayscale image (4x4)
    uint8_t image[width * height] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    // Output initial image
    cout << "Initial Greyscale Image:" << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cout << static_cast<int>(image[i * width + j]) << " ";
        }
        cout << endl;
    }

    // Cutoff frequency for the Gaussian High-Pass Filter
    double cutoff_frequency = 2.0;

    // Convert the grayscale image to a 2D complex array
    cuDoubleComplex** complex_image = convertUint8ToCuComplex2D(image, width, height);

    // Perform 2D FFT
    cout << "Performing 2D FFT..." << endl;
    cuDoubleComplex** fft_result = FFT2DParallel(complex_image, width, height);

    // Apply Gaussian High-Pass Filter
    cout << "Applying Gaussian High-Pass Filter..." << endl;
    cuDoubleComplex** filtered_result = gaussianHighPassFilterCUDA(fft_result, width, height, cutoff_frequency);

    // Perform Inverse FFT
    cout << "Performing Inverse FFT..." << endl;
    cuDoubleComplex** reconstructed_image = IFFT2DParallel(filtered_result, width, height);

    // Output reconstructed results
    cout << "Reconstructed Image:" << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cout << reconstructed_image[i][j].x << " ";
        }
        cout << endl;
    }

    // Cleanup dynamically allocated arrays
    for (int i = 0; i < height; ++i) {
        delete[] complex_image[i];
        delete[] fft_result[i];
        delete[] filtered_result[i];
        delete[] reconstructed_image[i];
    }
    delete[] complex_image;
    delete[] fft_result;
    delete[] filtered_result;
    delete[] reconstructed_image;

    cout << "Test completed successfully!" << endl;
}

void printTestGrayscaleCuComplex2DConversion() {
    bool testResults = testGrayscaleComplexConversion();
    if (testResults) {
        cout << "greyscale image conversion to and from cuComplex2D working." << endl;
    }
    else {
        cout << "greyscale image conversion to and from cuComplex2D not working." << endl;
    }
}

void printGaussianFilterTestResults() {
    bool testResutlts = testGaussianHighPassFilterCUDA();
    if (testResutlts) {
        cout << "Gaussian High pass filter working." << endl;
    }
    else {
        cout << "Gaussian High pass filter failed." << endl;
    }
}

void printFastFourierTransformTestResults() {
    /*bool testResults = testFFTAndIFFT();
    if (testResults) {
        cout << "Fast Fourier Transform working." << endl;
    }
    else {
        cout << "Fast Fourier Transform failed." << endl;
    }*/
    testFFTAndIFFT();
}

int testCUDA()
{
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        cerr << "CUDA error: " << cudaGetErrorString(error) << endl;
        return -1;
    }

    if (deviceCount == 0) {
        cout << "No CUDA-capable GPU detected." << endl;
    }
    else {
        cout << "Number of CUDA-capable GPUs: " << deviceCount << endl;
        for (int i = 0; i < deviceCount; i++) {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, i);
            cout << "GPU " << i << ": " << deviceProp.name << endl;
        }
    }

    return 0;
}

int checkGPUInformation() {
    // Get the number of CUDA devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices detected." << std::endl;
        return 1;
    }

    // Loop through devices and print properties
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "Device " << dev << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        std::cout << "  Maximum Threads per Block: " << prop.maxThreadsPerBlock << std::endl;

        if (prop.major >= 1) {
            std::cout << "  Double Precision Support: ";
            if (prop.major >= 2 || (prop.major == 1 && prop.minor >= 3)) {
                std::cout << "Yes" << std::endl;
            }
            else {
                std::cout << "No" << std::endl;
            }
        }
        else {
            std::cout << "  Double Precision Support: Not Supported" << std::endl;
        }
    }

    return 0;
}

int main()
{
    //testCUDA();
    //checkGPUInformation();
    //printGaussianFilterTestResults();
    //printFastFourierTransformTestResults();
    //printTestGrayscaleCuComplex2DConversion();
    testFFT2DToGaussianFilterToIFFT2D();
    //system("pause");
}
