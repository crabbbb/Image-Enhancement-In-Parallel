#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

using namespace std;

// Perform 1D FFT using the Cooley-Tukey algorithm
void FFT2D_inplace(complex<double>** data, int width, int height, int sign = +1);

void FFT1D_iterative(complex<double>* x, int n, int log2n, const vector<int>& bitRev, const vector< vector< complex<double> > >& twTable, int sign);

int getLog2(int n);

void prepareBitReversalTable(int n, int log2n, vector<int>& bitRev);

void prepareTwiddleTable(int n, int log2n, int sign, vector< vector< complex<double> > >& twiddleTable);

// Test function for FFT2D
int testFFT2D();

complex<double>** FFT2D(uint8_t* inputImage, int width, int height);

uint8_t* IFFT2D(complex<double>** data, int width, int height);



