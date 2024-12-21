#pragma once
#include <complex>

using namespace std;

// Function declarations
double computeHighPassValue(int u, int v, int height, int width, double cutoff_frequency);
void gaussianHighPassFilter(
    const complex<double>** F_shifted, // 2D input array
    complex<double>** G,              // 2D output array
    int width,
    int height,
    double cutoff_frequency);
bool testGaussianHighPassFilter();
