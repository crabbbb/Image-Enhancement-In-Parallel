#pragma once

#include <cmath>
#include <complex>
#include <cstdint>

using namespace std;

// Perform 1D FFT using the Cooley-Tukey algorithm
void FFT1D(complex<double>* x, int size);

// Perform 2D FFT by applying 1D FFT row-wise and column-wise
complex<double>** FFT2D(complex<double>** image, int width, int height);

// Test function for FFT2D
bool testFFT2D();



