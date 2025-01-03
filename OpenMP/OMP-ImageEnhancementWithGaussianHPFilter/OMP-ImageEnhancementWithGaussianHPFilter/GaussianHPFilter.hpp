#pragma once

#include <complex>

using namespace std;

void fftShift2D(std::complex<double>** data, int width, int height);

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

complex<double>** unsharpMaskingFrequencyDomain(
    complex<double>** F_unshifted, // Frequency data in "normal" layout (DC at top-left)
    int width,
    int height,
    double cutoff_frequency,
    double alpha);

// Test function for Gaussian High-Pass Filter
bool testGaussianHighPassFilter();

