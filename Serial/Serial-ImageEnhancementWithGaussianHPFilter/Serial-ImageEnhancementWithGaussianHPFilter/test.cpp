//// Serial-ImageEnhancementWithGaussianHPFilter.cpp : This file contains the 'main' function. Program execution begins and ends there.
////
//
//#include <iostream>
//#include "GaussianHPFilter.hpp"
//#include "FastFourierTransform.hpp"
//#include "InverseFastFourierTransform.hpp"
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
//        0, 255, 0, 255,
//        255, 0, 255, 0,
//        0, 255, 0, 255,
//        255, 0, 255, 0,
//	    0, 255, 0, 255
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
//    cout << endl;
//
//    // Cutoff frequency for the Gaussian High-Pass Filter
//    double cutoff_frequency = 128;
//    
//    // alpha for the unsharp masking
//    double alpha = 2;
//
//    // Zero-pad the image to power-of-two dimensions
//    int newWidth, newHeight;
//    uint8_t* padded_image = zeroPad2D(
//        image,  // original data
//        width,          // old width
//        height,         // old height
//        newWidth,       // [out] new width
//        newHeight       // [out] new height
//    );
//
//    // print the output after zero pad
//    cout << "Image (after zero pad) :" << endl;
//    for (int i = 0; i < newHeight; ++i) {
//        for (int j = 0; j < newWidth; ++j) {
//            cout << static_cast<int>(padded_image[i * width + j]) << " ";
//        }
//        cout << endl;
//    }
//
//    cout << endl;
//
//    // Perform 2D FFT
//    cout << "Performing 2D FFT..." << endl;
//    complex<double>** fft_result = FFT2D(padded_image, newWidth, newHeight);
//
//    // print complex number array after 2dfft
//    cout << "2D FFT complex number array:" << endl;
//    for (int i = 0; i < newHeight; ++i) {
//        for (int j = 0; j < newWidth; ++j) {
//            cout << "(" << fft_result[i][j].real() << ", " << fft_result[i][j].imag() << ") ";
//        }
//        cout << endl;
//    }
//
//    cout << endl;
//
//    // Apply Gaussian High-Pass Filter
//    cout << "Applying unsharp masking with Gaussian High-Pass Filter..." << endl;
//    complex<double>** filtered_result = unsharpMaskingFrequencyDomain(fft_result, newWidth, newHeight, cutoff_frequency, alpha);
//    cout << "complex number array after unsharp masking:" << endl;
//    for (int i = 0; i < newHeight; ++i) {
//        for (int j = 0; j < newWidth; ++j) {
//            cout << "(" << filtered_result[i][j].real() << ", " << filtered_result[i][j].imag() << ") ";
//        }
//        cout << endl;
//    }
//
//    cout << endl;
//
//    // Perform Inverse FFT
//    cout << "Performing Inverse FFT..." << endl;
//    uint8_t* reconstructed_image = IFFT2D(filtered_result, newWidth, newHeight);
//
//    uint8_t* reconstructed_image_with_padding_removed = unzeroPad2D(reconstructed_image, newWidth, newHeight, width, height);
//
//    // print the output before crop
//    cout << "Reconstructed Image (before crop) :" << endl;
//    for (int i = 0; i < newHeight; ++i) {
//        for (int j = 0; j < newWidth; ++j) {
//            cout << static_cast<int>(reconstructed_image[i * width + j]) << " ";
//        }
//        cout << endl;
//    }
//
//    cout << endl;
//
//    // crop back to original dimensions
//
//    // Output reconstructed results
//    cout << "Reconstructed Image:" << endl;
//    for (int i = 0; i < height; ++i) {
//        for (int j = 0; j < width; ++j) {
//            cout << static_cast<int>(reconstructed_image_with_padding_removed[i * width + j]) << " ";
//        }
//        cout << endl;
//    }
//
//    cout << endl;
//
//    // Cleanup dynamically allocated arrays
//    cleanup2DArray(fft_result, height);
//    cleanup2DArray(filtered_result, height);
//
//    cout << "Test completed successfully!" << endl;
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
//void printFastFourierTransformTestResults() {
//    bool testResutlts = testFFT2D();
//    if (testResutlts) {
//        cout << "Fast Fourier Transform working." << endl;
//    }
//    else {
//        cout << "Fast Fourier Transform failed." << endl;
//    }
//}
//
//void printInverseFastFourierTransformTestResults() {
//    bool testResutlts = testIFFT2D();
//    if (testResutlts) {
//        cout << "Inverse Fast Fourier Transform working." << endl;
//    }
//    else {
//        cout << "Inverse Fast Fourier Transform failed." << endl;
//    }
//}
//
//int main()
//{
//    //testConversionToAndFromComplex();
//    //printGaussianFilterTestResults();
//    //printFastFourierTransformTestResults();
//    //printInverseFastFourierTransformTestResults();
//    testFFT2DToGaussianFilterToIFFT2D();
//    return 0;
//}
