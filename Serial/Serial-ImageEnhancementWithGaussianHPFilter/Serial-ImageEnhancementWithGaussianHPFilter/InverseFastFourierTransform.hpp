#pragma once
#include <complex>

using namespace std;

complex<double>* IFFT1D(std::complex<double>* x, int size);
complex<double>** IFFT2D(std::complex<double>** image, int width, int height);
bool testIFFT2D();