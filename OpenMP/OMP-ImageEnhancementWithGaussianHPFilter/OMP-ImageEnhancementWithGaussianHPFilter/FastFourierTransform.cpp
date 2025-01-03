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

using namespace std;

//------------------------------------------------------------------------------
// In-place iterative FFT (Cooley-Tukey)
// x: pointer to array of complex samples
// n: size of the FFT (must be a power of 2)
// sign: +1 for forward FFT, -1 for inverse FFT
//------------------------------------------------------------------------------
void FFT1D_iterative(complex<double>* x, int n, int sign = +1)
{
    // 1) Bit-reversal permutation
    // We assume n is a power of 2.  For example n=8 (1000), log2(n)=3.
    int log2n = 0;
    while ((1 << log2n) < n) {
        log2n++;
    }

    // Bit-reverse the indices to re-arrange the sequence 
    for (int i = 0; i < n; ++i) {
        // Reverse bits of i
        // using unsigned to only store the 0 or positive value 
        unsigned int reversed = 0;
        unsigned int temp = i;

        // to get the location
        for (int b = 0; b < log2n; ++b) {
            reversed = (reversed << 1) | (temp & 1);
            temp >>= 1;
        }

        if (static_cast<int>(reversed) > i) {
            swap(x[i], x[reversed]);
        }
    }

    // 2) Iterative FFT
    // "m" is the size of the current FFT sub-problem: 2,4,8,...,n
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;          // Current FFT length = 2^s
        int m2 = m >> 1;         // Half of current FFT length

        // Precompute the "step" twiddle (a single primitive root of unity)
        double theta = sign * (-2.0 * M_PI / m);
        complex<double> wm(cos(theta), sin(theta));

        // We apply the butterfly from k=0 in strides of m
        // This is the main loop that you can parallelize if it is large enough
        // but typically you get better performance for large n only.
        for (int k = 0; k < n; k += m) {
            complex<double> w(1.0, 0.0); // Twiddle factor increment
            for (int j = 0; j < m2; j++) {
                complex<double> t = w * x[k + j + m2];
                complex<double> u = x[k + j];
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

// Perform 2D FFT by doing 1D FFT on rows, then 1D FFT on columns.
void FFT2D_inplace(complex<double>** data, int width, int height, int sign = +1)
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
        vector<complex<double>> colBuf(height);
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

complex<double>** FFT2D(uint8_t* inputImage, int width, int height) {
    complex<double>** data = storeUint8ToComplex2D(inputImage, width, height);
    FFT2D_inplace(data, width, height, +1);
    return data;
}

uint8_t* IFFT2D(complex<double>** data, int width, int height) {
    // Optionally, perform inverse 2D FFT (just pass sign = -1)]
    FFT2D_inplace(data, width, height, -1);
    return storeComplex2DToUint8(data, width, height);
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

    cout << "\nOriginal image:\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cout << static_cast<int>(image[i * width + j]) << " ";
        }
        cout << endl;
    }


    // Perform forward 2D FFT
    complex<double>** fft_results = FFT2D(image, width, height);

    // Print results
    cout << "Forward FFT2D Result:\n";
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            cout << fft_results[r][c] << " ";
        }
        cout << endl;
    }

    uint8_t* ifft_results = IFFT2D(fft_results, width, height);

    // Print results after inverse (should roughly match the original image)
    cout << "\nAfter Inverse FFT2D (should be close to original):\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cout << static_cast<int>(ifft_results[i * width + j]) << " ";
        }
        cout << endl;
    }

    // Cleanup
    cleanup2DArray(fft_results, height);
    return true;
}
