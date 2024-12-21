#include <cmath>
#include <complex>
#include <iostream>
#include "GaussianHPFilter.h"

using namespace std;

// Function to compute Gaussian High-Pass Filter for a given pixel
double computeHighPassValue(int u, int v, int height, int width, double cutoff_frequency) {
    // Calculate the distance D(u, v) from the center
    double D_uv = sqrt(pow(u - height / 2.0, 2) + pow(v - width / 2.0, 2));
    // Compute the Gaussian High-Pass Filter value H(u, v)
    return 1.0 - exp(-pow(D_uv, 2) / (2 * pow(cutoff_frequency, 2)));
}

// Serial implementation of Gaussian High-Pass Filter
void gaussianHighPassFilter(
    const complex<double>* F_shifted,
    complex<double>* G,
    int width,
    int height,
    double cutoff_frequency)
{
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            // Compute the filter value
            double H_uv = computeHighPassValue(u, v, height, width, cutoff_frequency);
            // Apply the filter
            G[u * width + v] = F_shifted[u * width + v] * H_uv;
        }
    }
}

bool testGaussianHighPassFilter() {
    // Image dimensions
    const int width = 4;
    const int height = 4;

    // Cutoff frequency
    double cutoff_frequency = 2.0;

    // Simulated frequency-domain data (4x4 matrix, flattened as a 1D array)
    complex<double> F_shifted[width * height] = {
        {1, 1}, {2, 2}, {3, 3}, {4, 4},
        {5, 5}, {6, 6}, {7, 7}, {8, 8},
        {9, 9}, {10, 10}, {11, 11}, {12, 12},
        {13, 13}, {14, 14}, {15, 15}, {16, 16}
    };

    // Expected output (manually calculated for this example)
    complex<double> expected[width * height];
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            double D_uv = sqrt(pow(u - height / 2.0, 2) + pow(v - width / 2.0, 2));
            double H_uv = 1.0 - exp(-pow(D_uv, 2) / (2 * pow(cutoff_frequency, 2)));
            expected[u * width + v] = F_shifted[u * width + v] * H_uv;
        }
    }

    // Output storage for the function result
    complex<double> G[width * height];

    // Call the Gaussian High-Pass Filter
    gaussianHighPassFilter(F_shifted, G, width, height, cutoff_frequency);

    // Compare the output with the expected result
    bool test_passed = true;
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            // Allow for small floating-point differences
            if (abs(G[u * width + v].real() - expected[u * width + v].real()) > 1e-6 ||
                abs(G[u * width + v].imag() - expected[u * width + v].imag()) > 1e-6) {
                test_passed = false;
                cout << "Mismatch at (" << u << ", " << v << "): "
                    << "Expected (" << expected[u * width + v].real() << ", " << expected[u * width + v].imag()
                    << "), Got (" << G[u * width + v].real() << ", " << G[u * width + v].imag() << ")" << endl;
            }
        }
    }

    return test_passed;
}