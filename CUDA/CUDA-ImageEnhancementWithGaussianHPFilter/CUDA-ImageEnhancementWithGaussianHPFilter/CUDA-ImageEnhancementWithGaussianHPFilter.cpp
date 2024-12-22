// CUDA-ImageEnhancementWithGaussianHPFilter.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cuda_runtime.h>
#include "CUDA-GaussianHPFIlter.cuh"
#include "CUDA-FastFourierTransform.cuh"

using namespace std;

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
    bool testResutlts = testFFT2D();
    if (testResutlts) {
        cout << "Fast Fourier Transform working." << endl;
    }
    else {
        cout << "Fast Fourier Transform failed." << endl;
    }
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
    printFastFourierTransformTestResults();

    //system("pause");
}
