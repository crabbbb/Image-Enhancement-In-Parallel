#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cstdint>
#include <cassert>
#include "Utils.hpp"

// ---------------------------------------------------------------------------
// If M_PI is not defined, define it:
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

// ---------------------------------------------------------------------------
// Precompute bit-reversed indices for an array of size n = 2^log2n
// bitRev[i] = the bit-reversed index of i
// ---------------------------------------------------------------------------
void prepareBitReversalTable(int n, int log2n, vector<int>& bitRev)
{
    bitRev.resize(n);
    for (int i = 0; i < n; i++) {
        int j = 0;
        int x = i;
        // Reverse bits in x
        for (int b = 0; b < log2n; b++) {
            j = (j << 1) | (x & 1);
            x >>= 1;
        }
        bitRev[i] = j;
    }
}

// ---------------------------------------------------------------------------
// Precompute complex twiddle factors for each "stage" in the iterative FFT
// We store twiddles in a 2D array: twiddleTable[stage][k]
// stage ranges 1..log2n
// k ranges 0..(1<<stage)-1, but effectively we only use 0..(m2-1) 
//   for each stage, with m2 = 1<<(stage-1).
// ---------------------------------------------------------------------------
void prepareTwiddleTable(int n, int log2n, int sign, vector< vector< complex<double> > >& twiddleTable)
{
    twiddleTable.resize(log2n + 1);  // stage 1..log2n

    // For each stage s, we have "m = 1<<s"
    // and we compute the base twiddle factor w_m = exp(sign * -2pi i / m)
    // then we generate all powers of w_m from 0..m2-1
    // w^k for k = 0..m2-1, where m2 = m/2.
    for (int s = 1; s <= log2n; s++) {
        int m = (1 << s);
        int m2 = (m >> 1);

        twiddleTable[s].resize(m2);

        double theta = sign * -2.0 * M_PI / double(m);
        complex<double> wm(cos(theta), sin(theta));

        // Precompute the progression w^0, w^1, ..., w^(m2-1)
        complex<double> w(1.0, 0.0);
        for (int k = 0; k < m2; k++) {
            twiddleTable[s][k] = w;
            w *= wm;
        }
    }
}

// ---------------------------------------------------------------------------
// 1D In-place Iterative FFT using precomputed bitReverse and twiddleTable
//    x       : pointer to array of complex samples
//    n       : size of the FFT (must be a power of 2)
//    log2n   : log2(n)
//    bitRev  : bit-reversed indices array
//    twTable : twiddleTable[stage][k]
//    sign    : +1 for forward FFT, -1 for inverse FFT
// ---------------------------------------------------------------------------
void FFT1D_iterative(complex<double>* x, int n, int log2n, const vector<int>& bitRev, const vector< vector< complex<double> > >& twTable, int sign)
{
    // 1) Bit-reversal permutation (swap x[i] with x[bitRev[i]] for i < bitRev[i])
    for (int i = 0; i < n; i++) {
        int j = bitRev[i];
        if (j > i) {
            swap(x[i], x[j]);
        }
    }

    // 2) Iterative FFT passes
    for (int s = 1; s <= log2n; s++) {
        int m = 1 << s;     // size of the sub-FFT
        int m2 = m >> 1;     // half of sub-FFT

        // twTable[s][k] = w^k for that stage's base twiddle
        const auto& twiddles = twTable[s];

        // For each sub-block of size m
        for (int k = 0; k < n; k += m) {
            // For each element in the lower half of that sub-block
            for (int j = 0; j < m2; j++) {
                // "rotated" element
                complex<double> t = twiddles[j] * x[k + j + m2];
                // "unrotated" element
                complex<double> u = x[k + j];
                // butterfly
                x[k + j] = u + t;
                x[k + j + m2] = u - t;
            }
        }
    }

    // 3) For inverse transform, divide by n
    if (sign == -1) {
        for (int i = 0; i < n; i++) {
            x[i] /= double(n);
        }
    }
}

// Helper to compute integer log2(n). (Assumes n is a power of 2.)
int getLog2(int n) {
    int log2n = 0;
    while ((1 << log2n) < n) {
        log2n++;
    }
    return log2n;
}

// ---------------------------------------------------------------------------
// 2D FFT: do 1D FFT on rows, then 1D FFT on columns
//         using precomputed tables for "width" and "height".
// ---------------------------------------------------------------------------
void FFT2D_inplace(complex<double>** data, int width, int height, int sign)
{
    // Precompute for row size
    int log2W = getLog2(width);
    vector<int> bitRevW;
    vector< vector< complex<double> > > twiddleTableW;
    prepareBitReversalTable(width, log2W, bitRevW);
    prepareTwiddleTable(width, log2W, sign, twiddleTableW);

    // 1) FFT each row
    for (int r = 0; r < height; r++) {
        FFT1D_iterative(data[r], width, log2W, bitRevW, twiddleTableW, sign);
    }

    // Precompute for column size
    int log2H = getLog2(height);
    vector<int> bitRevH;
    vector< vector< complex<double> > > twiddleTableH;
    prepareBitReversalTable(height, log2H, bitRevH);
    prepareTwiddleTable(height, log2H, sign, twiddleTableH);

    // 2) FFT each column
    vector<complex<double>> colBuf(height);
    for (int c = 0; c < width; c++) {
        // Copy column c into temp
        for (int r = 0; r < height; r++) {
            colBuf[r] = data[r][c];
        }
        // Perform 1D FFT on colBuf
        FFT1D_iterative(colBuf.data(), height, log2H, bitRevH, twiddleTableH, sign);
        // Store back
        for (int r = 0; r < height; r++) {
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

// ---------------------------------------------------------------------------
// Small demonstration: 2D FFT of a 4x4 block
// ---------------------------------------------------------------------------
int testFFT2D()
{
    const int width = 4;
    const int height = 4;

    // Allocate test data
    complex<double>** data = new complex<double>*[height];
    for (int r = 0; r < height; r++) {
        data[r] = new complex<double>[width];
        for (int c = 0; c < width; c++) {
            // Fill with something easy to see
            double val = double(r * width + c + 1);
            data[r][c] = complex<double>(val, 0.0);
        }
    }

    // Print original
    cout << "Original:\n";
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            cout << data[r][c].real() << " ";
        }
        cout << "\n";
    }

    // Forward FFT
    FFT2D_inplace(data, width, height, +1);

    cout << "\nAfter Forward 2D FFT:\n";
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            cout << "(" << data[r][c].real() << "," << data[r][c].imag() << ") ";
        }
        cout << "\n";
    }

    // Inverse FFT
    FFT2D_inplace(data, width, height, -1);

    cout << "\nAfter Inverse 2D FFT (should match original real parts):\n";
    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            // Round or cast if needed
            double val = data[r][c].real();
            cout << val << " ";
        }
        cout << "\n";
    }

    // Cleanup
    for (int r = 0; r < height; r++) {
        delete[] data[r];
    }
    delete[] data;

    return 0;
}
