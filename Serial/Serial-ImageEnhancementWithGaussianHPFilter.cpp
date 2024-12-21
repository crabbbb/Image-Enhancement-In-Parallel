// Serial-ImageEnhancementWithGaussianHPFilter.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "GaussianHPFilter.hpp"
#include "FastFourierTransform.hpp"
#include "InverseFastFourierTransform.hpp"
#include "Utils.hpp"

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
    complex<double>** complex_image = convertToComplex2D(image, width, height);

    // Perform 2D FFT
    cout << "Performing 2D FFT..." << endl;
    complex<double>** fft_result = FFT2D(complex_image, width, height);

    // Apply Gaussian High-Pass Filter
    cout << "Applying Gaussian High-Pass Filter..." << endl;
    complex<double>** filtered_result = gaussianHighPassFilter(fft_result, width, height, cutoff_frequency);

    // Perform Inverse FFT
    cout << "Performing Inverse FFT..." << endl;
    complex<double>** reconstructed_image = IFFT2D(filtered_result, width, height);

    // Output reconstructed results
    cout << "Reconstructed Image:" << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cout << reconstructed_image[i][j].real() << " ";
        }
        cout << endl;
    }

    // Cleanup dynamically allocated arrays
    cleanup2DArray(complex_image, height);
    cleanup2DArray(fft_result, height);
    cleanup2DArray(filtered_result, height);
    cleanup2DArray(reconstructed_image, height);

    cout << "Test completed successfully!" << endl;
}

void printGaussianFilterTestResults() {
    bool testResutlts = testGaussianHighPassFilter();
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

void printInverseFastFourierTransformTestResults() {
    bool testResutlts = testIFFT2D();
    if (testResutlts) {
        cout << "Inverse Fast Fourier Transform working." << endl;
    }
    else {
        cout << "Inverse Fast Fourier Transform failed." << endl;
    }
}

int main()
{
    //testConversionToAndFromComplex();
    //printGaussianFilterTestResults();
    //printFastFourierTransformTestResults();
    //printInverseFastFourierTransformTestResults();
    //testFFT2DToGaussianFilterToIFFT2D();
    return 0;
}
