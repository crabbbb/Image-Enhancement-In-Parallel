//// CUDA-ImageEnhancementWithGaussianHPFilter.cpp : This file contains the 'main' function. Program execution begins and ends there.
////
//
//#include <iostream>
//#include <cuda_runtime.h>
//#include <cuComplex.h>
//#include "GaussianHPFIlter.cuh"
//#include "FastFourierTransform.cuh"
//#include "Utils.hpp"
//
//using namespace std;
//
//void testFFT2DToGaussianFilterToIFFT2D() {
//    // Image dimensions
//    const int width = 4;
//    const int height = 5;
//
//    // Simulated grayscale image (4x4)
//    uint8_t image[width * height] = {
//        1, 2, 3, 4,
//        5, 6, 7, 8,
//        9, 10, 11, 12,
//        13, 14, 15, 16,
//        13, 14, 15, 16,
//    };
//
//    // Output initial image
//    cout << "Initial Greyscale Image:" << endl;
//    for (int i = 0; i < height; ++i) {
//        for (int j = 0; j < width; ++j) {
//            cout << static_cast<int>(image[i * width + j]) << " ";
//        }
//        cout << endl;
//    }
//
//    // Cutoff frequency for the Gaussian High-Pass Filter
//    double cutoff_frequency = 100.0;
//
//    double alpha = 1.0;
//
//    // Convert the grayscale image to a 2D complex array
//    cuDoubleComplex** complex_image = convertUint8ToCuComplex2D(image, width, height);
//
//    // Zero-pad the image to power-of-two dimensions
//    int newWidth, newHeight;
//    cuDoubleComplex** padded_complex_image = zeroPad2D(
//        complex_image,  // original data
//        width,          // old width
//        height,         // old height
//        newWidth,       // [out] new width
//        newHeight       // [out] new height
//    );
//
//    // Cleanup original complex_image since we no longer need it
//    cleanup2DArray(complex_image, height);
//
//    // Perform 2D FFT
//    cout << "Performing 2D FFT..." << endl;
//    cuDoubleComplex** fft_result = FFT2DParallel(padded_complex_image, newWidth, newHeight);
//
//    // Apply Gaussian High-Pass Filter
//    cout << "Applying Gaussian High-Pass Filter..." << endl;
//    cuDoubleComplex** filtered_result = unsharpMaskingFrequencyDomain(fft_result, newWidth, newHeight, cutoff_frequency, alpha);
//
//    // Perform Inverse FFT
//    cout << "Performing Inverse FFT..." << endl;
//    cuDoubleComplex** reconstructed_image = IFFT2DParallel(filtered_result, newWidth, newHeight);
//
//    cuDoubleComplex** reconstructed_image_with_padding_removed = unzeroPad2D(reconstructed_image, newWidth, newHeight, width, height);
//
//    // Output reconstructed results
//    cout << "Reconstructed Image:" << endl;
//    for (int i = 0; i < height; ++i) {
//        for (int j = 0; j < width; ++j) {
//            cout << reconstructed_image_with_padding_removed[i][j].x << " ";
//        }
//        cout << endl;
//    }
//
//    // Cleanup dynamically allocated arrays
//    cleanup2DArray(padded_complex_image, newHeight);
//    cleanup2DArray(fft_result, newHeight);
//    cleanup2DArray(filtered_result, newHeight);
//    cleanup2DArray(reconstructed_image, newHeight);
//    cleanup2DArray(reconstructed_image_with_padding_removed, height);
//
//    cout << "Test completed successfully!" << endl;
//}
//
//void printTestGrayscaleCuComplex2DConversion() {
//    bool testResults = testGrayscaleComplexConversion();
//    if (testResults) {
//        cout << "greyscale image conversion to and from cuComplex2D working." << endl;
//    }
//    else {
//        cout << "greyscale image conversion to and from cuComplex2D not working." << endl;
//    }
//}
//
//void printGaussianFilterTestResults() {
//    bool testResutlts = testUnsharpMasking();
//    if (testResutlts) {
//        cout << "Gaussian High pass filter working." << endl;
//    }
//    else {
//        cout << "Gaussian High pass filter failed." << endl;
//    }
//}
//
////void printFastFourierTransformTestResults() {
////    /*bool testResults = testFFTAndIFFT();
////    if (testResults) {
////        cout << "Fast Fourier Transform working." << endl;
////    }
////    else {
////        cout << "Fast Fourier Transform failed." << endl;
////    }*/
////    testFFTAndIFFT();
////}
//
//int testCUDA()
//{
//    int deviceCount = 0;
//    cudaError_t error = cudaGetDeviceCount(&deviceCount);
//
//    if (error != cudaSuccess) {
//        cerr << "CUDA error: " << cudaGetErrorString(error) << endl;
//        return -1;
//    }
//
//    if (deviceCount == 0) {
//        cout << "No CUDA-capable GPU detected." << endl;
//    }
//    else {
//        cout << "Number of CUDA-capable GPUs: " << deviceCount << endl;
//        for (int i = 0; i < deviceCount; i++) {
//            cudaDeviceProp deviceProp;
//            cudaGetDeviceProperties(&deviceProp, i);
//            cout << "GPU " << i << ": " << deviceProp.name << endl;
//        }
//    }
//
//    return 0;
//}
//
//int checkGPUInformation() {
//    // Get the number of CUDA devices
//    int deviceCount;
//    cudaGetDeviceCount(&deviceCount);
//    if (deviceCount == 0) {
//        std::cout << "No CUDA-capable devices detected." << std::endl;
//        return 1;
//    }
//
//    // Loop through devices and print properties
//    for (int dev = 0; dev < deviceCount; ++dev) {
//        cudaDeviceProp prop;
//        cudaGetDeviceProperties(&prop, dev);
//
//        std::cout << "Device " << dev << ": " << prop.name << std::endl;
//        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
//        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
//        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
//        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
//        std::cout << "  Maximum Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
//
//        if (prop.major >= 1) {
//            std::cout << "  Double Precision Support: ";
//            if (prop.major >= 2 || (prop.major == 1 && prop.minor >= 3)) {
//                std::cout << "Yes" << std::endl;
//            }
//            else {
//                std::cout << "No" << std::endl;
//            }
//        }
//        else {
//            std::cout << "  Double Precision Support: Not Supported" << std::endl;
//        }
//    }
//
//    return 0;
//}
//
//int main()
//{
//    //testCUDA();
//    //checkGPUInformation();
//    //printGaussianFilterTestResults();
//    //printFastFourierTransformTestResults();
//    //printTestGrayscaleCuComplex2DConversion();
//    testFFT2DToGaussianFilterToIFFT2D();
//    //system("pause");
//}
