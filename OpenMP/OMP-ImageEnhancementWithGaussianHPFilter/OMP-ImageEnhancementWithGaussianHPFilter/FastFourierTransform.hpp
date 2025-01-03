#pragma once

#include <cmath>
#include <complex>
#include <cstdint>

using namespace std;

// Perform 1D FFT using the Cooley-Tukey algorithm
void FFT1D_iterative(std::complex<double>* x, int n, int sign = +1);

void FFT2D_inplace(std::complex<double>** data, int width, int height, int sign = +1);

// Test function for FFT2D
bool testFFT2D();

complex<double>** FFT2D(uint8_t* inputImage, int width, int height);

uint8_t* IFFT2D(complex<double>** data, int width, int height);



