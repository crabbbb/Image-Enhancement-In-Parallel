#pragma once

#include <cmath>
#include <complex>
#include <cstdint>

using namespace std;

// Perform 1D FFT using the Cooley-Tukey algorithm
void FFT1D_iterative(std::complex<double>* x, int n, int sign = +1);

// Perform 2D FFT by applying 1D FFT row-wise and column-wise
std::complex<double>** allocate2DComplex(int height, int width);

void cleanup2DArray(std::complex<double>** arr, int height);

std::complex<double>** convertUint8ToComplex2D(const uint8_t* image, int width, int height);

void FFT2D_inplace(std::complex<double>** data, int width, int height, int sign = +1);

// Test function for FFT2D
bool testFFT2D();



