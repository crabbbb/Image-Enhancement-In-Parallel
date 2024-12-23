#include <cmath>
#include <complex>
#include <vector>
#include <cstdint>
#include <omp.h>
#include <iostream>
#include "Utils.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//------------------------------------------------------------------------------
// In-place iterative FFT (Cooley-Tukey)
// x: pointer to array of complex samples
// n: size of the FFT (must be a power of 2)
// sign: +1 for forward FFT, -1 for inverse FFT
//------------------------------------------------------------------------------
void FFT1D_iterative(std::complex<double>* x, int n, int sign = +1)
{
    // 1) Bit-reversal permutation
    // We assume n is a power of 2.  For example n=8 (1000), log2(n)=3.
    int log2n = 0;
    while ((1 << log2n) < n) {
        log2n++;
    }

    // Bit-reverse the indices
    for (int i = 0; i < n; ++i) {
        // Reverse bits of i
        unsigned int reversed = 0;
        unsigned int temp = i;
        for (int b = 0; b < log2n; ++b) {
            reversed = (reversed << 1) | (temp & 1);
            temp >>= 1;
        }
        if (static_cast<int>(reversed) > i) {
            std::swap(x[i], x[reversed]);
        }
    }

    // 2) Iterative FFT
    // "m" is the size of the current FFT sub-problem: 2,4,8,...,n
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;          // Current FFT length = 2^s
        int m2 = m >> 1;         // Half of current FFT length

        // Precompute the "step" twiddle (a single primitive root of unity)
        double theta = sign * (-2.0 * M_PI / m);
        std::complex<double> wm(std::cos(theta), std::sin(theta));

        // We apply the butterfly from k=0 in strides of m
        // This is the main loop that you can parallelize if it is large enough
        // but typically you get better performance for large n only.
        for (int k = 0; k < n; k += m) {
            std::complex<double> w(1.0, 0.0); // Twiddle factor increment
            for (int j = 0; j < m2; j++) {
                std::complex<double> t = w * x[k + j + m2];
                std::complex<double> u = x[k + j];
                x[k + j] = u + t;
                x[k + j + m2] = u - t;
                w *= wm;
            }
        }
    }

    // If inverse FFT, we need to divide each element by n
    if (sign == -1) {
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            x[i] /= static_cast<double>(n);
        }
    }
}

std::complex<double>** allocate2DComplex(int height, int width) {
    auto array2D = new std::complex<double>*[height];
    for (int i = 0; i < height; ++i) {
        array2D[i] = new std::complex<double>[width];
    }
    return array2D;
}

// Allocate a 2D array of complex<double>
// convert uint8_t* grayscale image to a 2D array of complex numbers
std::complex<double>** convertToComplex2D(const uint8_t* image, int width, int height) {
    std::complex<double>** out = allocate2DComplex(height, width);
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            out[r][c] = std::complex<double>(static_cast<double>(image[r * width + c]), 0.0);
        }
    }
    return out;
}

// Cleanup
void cleanup2DArray(std::complex<double>** arr, int height) {
    for (int i = 0; i < height; ++i) {
        delete[] arr[i];
    }
    delete[] arr;
}

// Perform 2D FFT by doing 1D FFT on rows, then 1D FFT on columns.
void FFT2D_inplace(std::complex<double>** data, int width, int height, int sign = +1)
{
    // 1) FFT each row
    #pragma omp parallel for
    for (int r = 0; r < height; ++r) {
        FFT1D_iterative(data[r], width, sign);
    }

    // 2) FFT each column
    // For column FFT, gather each column into a temp buffer, call FFT, then store back.
    // You can parallelize across columns as well.
    #pragma omp parallel for
    for (int c = 0; c < width; ++c) {
        // Copy column 'c' into temp
        std::vector<std::complex<double>> colBuf(height);
        for (int r = 0; r < height; ++r) {
            colBuf[r] = data[r][c];
        }
        // Perform 1D FFT on colBuf
        FFT1D_iterative(colBuf.data(), height, sign);

        // Store back
        for (int r = 0; r < height; ++r) {
            data[r][c] = colBuf[r];
        }
    }
}

bool testFFT2D() {
    const int width = 4;
    const int height = 4;
    uint8_t image[width * height] = {
         1,  2,  3,  4,
         5,  6,  7,  8,
         9, 10, 11, 12,
        13, 14, 15, 16
    };

    // Convert the grayscale image to a 2D array of complex numbers
    std::complex<double>** complex_image = convertToComplex2D(image, width, height);

    // Perform forward 2D FFT in-place
    FFT2D_inplace(complex_image, width, height, +1);

    // Print results
    std::cout << "Forward FFT2D Result:\n";
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            std::cout << complex_image[r][c] << " ";
        }
        std::cout << std::endl;
    }

    // Optionally, perform inverse 2D FFT (just pass sign = -1)
    FFT2D_inplace(complex_image, width, height, -1);

    // Print results after inverse (should roughly match the original image)
    std::cout << "\nAfter Inverse FFT2D (should be close to original):\n";
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            std::cout << complex_image[r][c] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    cleanup2DArray(complex_image, height);
    return true;
}
