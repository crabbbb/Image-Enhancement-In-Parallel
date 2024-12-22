// CUDA-ImageEnhancementWithGaussianHPFilter.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <cuda_runtime.h>
#include "CUDA-GaussianHPFIlter.cuh"

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

int main()
{
    //testCUDA();
    //printGaussianFilterTestResults();
}
