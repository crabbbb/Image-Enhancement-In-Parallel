#pragma once

#include <complex>

using namespace std;

// Function to compute Gaussian High-Pass Filter for a given pixel
double computeHighPassValue(int u, int v, int height, int width, double cutoff_frequency);

// Serial implementation of Gaussian High-Pass Filter
// Applies the filter to the frequency-domain representation of an image
// Returns the filtered output as a 2D array of complex<double>
complex<double>** gaussianHighPassFilter(
    complex<double>** F_shifted,
    int width,
    int height,
    double cutoff_frequency);

// Test function for Gaussian High-Pass Filter
bool testGaussianHighPassFilter();

