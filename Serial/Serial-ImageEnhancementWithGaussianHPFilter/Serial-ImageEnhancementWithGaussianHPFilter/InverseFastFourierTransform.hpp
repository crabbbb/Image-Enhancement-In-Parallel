#pragma once
#include <complex>

using namespace std;

complex<double>* IFFT1D(complex<double>* x, int size);
uint8_t* IFFT2D(complex<double>** image, int width, int height);
bool testIFFT2D();