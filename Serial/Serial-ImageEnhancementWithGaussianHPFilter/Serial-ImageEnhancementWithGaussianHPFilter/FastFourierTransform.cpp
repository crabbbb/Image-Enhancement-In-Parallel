#include <cmath>
#include <complex>
#include <iostream>
#include <cstdint>
#include "Utils.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

// Perform 1D FFT using the Cooley-Tukey algorithm
void FFT1D(complex<double>* x, int size) {
    // Base case
    if (size <= 1) return;

    // Split the array into even and odd parts
    complex<double>* x_even = new complex<double>[size / 2];
    complex<double>* x_odd = new complex<double>[size / 2];

    for (int i = 0; i < size / 2; ++i) {
        x_even[i] = x[i * 2];
        x_odd[i] = x[i * 2 + 1];
    }

    // Recursive FFT on even and odd parts
    FFT1D(x_even, size / 2);
    FFT1D(x_odd, size / 2);

    // Combine results
    for (int k = 0; k < size / 2; ++k) {
        complex<double> W_k = exp(complex<double>(0, -2.0 * M_PI * k / size)) * x_odd[k];
        x[k] = x_even[k] + W_k;
        x[k + size / 2] = x_even[k] - W_k;
    }

    delete[] x_even;
    delete[] x_odd;
}

// Perform 2D FFT by applying 1D FFT row-wise and column-wise
// Returns a new 2D array containing the FFT result.
complex<double>** FFT2D(const uint8_t* image, int width, int height) {
    // Step 1: Create a new array to store the FFT result
    complex<double>** fft_result = storeUint8ToComplex2D(image, width, height);

    // Step 2: Apply 1D FFT row-wise on the new array
    for (int i = 0; i < height; ++i) {
        FFT1D(fft_result[i], width);
    }

    // Step 3: Apply 1D FFT column-wise on the new array
    complex<double>* column = new complex<double>[height];
    for (int j = 0; j < width; ++j) {
        for (int i = 0; i < height; ++i) {
            column[i] = fft_result[i][j];
        }

        FFT1D(column, height);

        for (int i = 0; i < height; ++i) {
            fft_result[i][j] = column[i];
        }
    }
    delete[] column;

    // Step 4: Return the newly created FFT result array
    return fft_result;
}

// Test function for FFT2D
bool testFFT2D() {
    // Example grayscale image represented as uint8_t*
    const int width = 4;
    const int height = 5;
    uint8_t image[width * height] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
    };

    // Zero-pad the image to power-of-two dimensions
    int newWidth, newHeight;
    uint8_t* padded_image = zeroPad2D(
        image,  // original data
        width,          // old width
        height,         // old height
        newWidth,       // [out] new width
        newHeight       // [out] new height
    );

    // Perform 2D FFT
    complex<double>** fft_result = FFT2D(padded_image, newWidth, newHeight);

    // Output FFT result
    cout << "FFT2D Result:" << endl;
    for (int i = 0; i < newHeight; ++i) {
        for (int j = 0; j < newWidth; ++j) {
            cout << fft_result[i][j] << " ";
        }
        cout << endl;
    }

    // Cleanup
    cleanup2DArray(fft_result, newHeight);

    return true; // Placeholder: Add actual validation if needed
}
