#include "InverseFastFourierTransform.hpp"
#include "FastFourierTransform.hpp"
#include "Utils.hpp"
#include <cmath>
#include <complex>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

// 1D Inverse Fast Fourier Transform
complex<double>* IFFT1D(complex<double>* x, int size) {
    if (size <= 1) return x;

    // Allocate arrays for even and odd parts
    complex<double>* x_even = new complex<double>[size / 2];
    complex<double>* x_odd = new complex<double>[size / 2];

    for (int i = 0; i < size / 2; ++i) {
        x_even[i] = x[i * 2];
        x_odd[i] = x[i * 2 + 1];
    }

    // Recursive IFFT calls
    x_even = IFFT1D(x_even, size / 2);
    x_odd = IFFT1D(x_odd, size / 2);

    // Combine results using twiddle factors
    for (int k = 0; k < size / 2; ++k) {
        complex<double> W_k = exp(complex<double>(0, 2.0 * M_PI * k / size)) * x_odd[k];
        x[k] = x_even[k] + W_k;
        x[k + size / 2] = x_even[k] - W_k;
    }

    //// Normalize the result
    //for (int k = 0; k < size; ++k) {
    //    x[k] /= size;
    //}

    delete[] x_even;
    delete[] x_odd;

    return x;
}

// 2D Inverse Fast Fourier Transform
uint8_t* IFFT2D(complex<double>** image, int width, int height) {
    // Create a new array to store the IFFT result
    complex<double>** ifft_result = new complex<double>*[height];
    for (int i = 0; i < height; ++i) {
        ifft_result[i] = new complex<double>[width];
        for (int j = 0; j < width; ++j) {
            ifft_result[i][j] = image[i][j];
        }
    }

    // Apply 1D IFFT row-wise
    for (int i = 0; i < height; ++i) {
        ifft_result[i] = IFFT1D(ifft_result[i], width);
    }

    // Apply 1D IFFT column-wise
    complex<double>* column = new complex<double>[height];
    for (int j = 0; j < width; ++j) {
        for (int i = 0; i < height; ++i) {
            column[i] = ifft_result[i][j];
        }

        column = IFFT1D(column, height);

        for (int i = 0; i < height; ++i) {
            ifft_result[i][j] = column[i];
        }
    }
    delete[] column;

    // Normalize the entire result
    double normalization_factor = width * height;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            ifft_result[i][j] /= normalization_factor;
        }
    }

    uint8_t* resultImage = storeComplex2DToUint8(ifft_result, width, height);

    return resultImage;
}

// Test function for IFFT2D
bool testIFFT2D() {
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

    // display original image
    cout << "original image Result:" << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cout << static_cast<int>(image[i * width + j]) << " ";
        }
        cout << endl;
    }

    // zero pad image
    int newWidth, newHeight;
    uint8_t* padded_image = zeroPad2D(image, width, height, newWidth, newHeight);

    // display zero padded image
    cout << "zero padded image Result:" << endl;
    for (int i = 0; i < newHeight; ++i) {
        for (int j = 0; j < newWidth; ++j) {
            cout << static_cast<int>(padded_image[i * width + j]) << " ";
        }
        cout << endl;
    }

    // Perform 2D FFT
    complex<double>** fft_result = FFT2D(padded_image, newWidth, newHeight);

    // Perform 2D IFFT
    uint8_t* ifft_result = IFFT2D(fft_result, newWidth, newHeight);

    // display zero padded image
    cout << "final result after ifft before padding removed:" << endl;
    for (int i = 0; i < newHeight; ++i) {
        for (int j = 0; j < newWidth; ++j) {
            cout << static_cast<int>(ifft_result[i * width + j]) << " ";
        }
        cout << endl;
    }

    uint8_t* ifft_result_with_padding_removed = unzeroPad2D(ifft_result, newWidth, newHeight, width, height);

    // Output IFFT result (real parts of the reconstructed image)
    cout << "IFFT2D (Reconstructed Image) Result:" << endl;
    for (int i = 0; i < newHeight; ++i) {
        for (int j = 0; j < newWidth; ++j) {
            cout << static_cast<int>(ifft_result_with_padding_removed[i * width + j]) << " ";
        }
        cout << endl;
    }

    // Cleanup
    cleanup2DArray(fft_result, height);

    return true;
}

