#pragma once
#include <complex>

// Function declarations
double computeHighPassValue(int u, int v, int height, int width, double cutoff_frequency);
void gaussianHighPassFilter(const std::complex<double>* F_shifted, std::complex<double>* G, int width, int height, double cutoff_frequency);
bool testGaussianHighPassFilter();
