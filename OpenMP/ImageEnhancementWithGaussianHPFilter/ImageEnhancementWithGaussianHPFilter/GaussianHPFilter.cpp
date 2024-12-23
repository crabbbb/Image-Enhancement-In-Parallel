#include <cmath>
#include <complex>
#include <iostream>
#include <omp.h>
#include "GaussianHPFilter.hpp"

using namespace std;

// Function to compute Gaussian High-Pass Filter for a given pixel
double computeHighPassValue(int u, int v, int height, int width, double cutoff_frequency) {
    // Calculate the distance D(u, v) from the center
    double D_uv = sqrt(pow(u - height / 2.0, 2) + pow(v - width / 2.0, 2));
    // Compute the Gaussian High-Pass Filter value H(u, v)
    return 1.0 - exp(-pow(D_uv, 2) / (2 * pow(cutoff_frequency, 2)));
}

// Serial implementation of Gaussian High-Pass Filter
complex<double>** gaussianHighPassFilter(
    complex<double>** F_shifted, // 2D input array
    int width,
    int height,
    double cutoff_frequency)
{
    // Create a 2D array for H(u, v)
    double** H = new double* [height];
    for (int i = 0; i < height; ++i) {
        H[i] = new double[width];
    }

    // Step 1: Compute the filter values H(u, v)
    // parallel over both dimension
    #pragma omp parallel for collapse(2)
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            H[u][v] = computeHighPassValue(u, v, height, width, cutoff_frequency);
        }
    }

    // Create the output array G
    complex<double>** G = new complex<double>*[height];
    for (int i = 0; i < height; ++i) {
        G[i] = new complex<double>[width];
    }

    // Step 2: Apply the filter
    #pragma omp parallel for collapse(2) 
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            G[u][v] = F_shifted[u][v] * H[u][v];
        }
    }

    // Cleanup the H array
    for (int i = 0; i < height; ++i) {
        delete[] H[i];
    }
    delete[] H;

    // Return the filtered output array
    return G;
}

bool testGaussianHighPassFilter() {
    // Image dimensions
    const int width = 4;
    const int height = 4;

    // Cutoff frequency
    double cutoff_frequency = 100.0;

    // Simulated frequency-domain data (4x4 matrix)
    complex<double>** F_shifted = new complex<double>*[height];
    for (int i = 0; i < height; ++i) {
        F_shifted[i] = new complex<double>[width];
    }

    int value = 1;
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            const_cast<complex<double>&>(F_shifted[u][v]) = complex<double>(value, value);
            ++value;
        }
    }

    // Expected output (manually calculated for this example)
    complex<double>** expected = new complex<double>*[height];
    for (int i = 0; i < height; ++i) {
        expected[i] = new complex<double>[width];
    }

    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            double D_uv = sqrt(pow(u - height / 2.0, 2) + pow(v - width / 2.0, 2));
            double H_uv = 1.0 - exp(-pow(D_uv, 2) / (2 * pow(cutoff_frequency, 2)));
            expected[u][v] = F_shifted[u][v] * H_uv;
        }
    }

    // Call the Gaussian High-Pass Filter and get the output
    complex<double>** G = gaussianHighPassFilter(F_shifted, width, height, cutoff_frequency);

    // Compare the output with the expected result
    bool test_passed = true;
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            // Allow for small floating-point differences
            if (abs(G[u][v].real() - expected[u][v].real()) > 1e-6 ||
                abs(G[u][v].imag() - expected[u][v].imag()) > 1e-6) {
                test_passed = false;
                cout << "Mismatch at (" << u << ", " << v << "): "
                    << "Expected (" << expected[u][v].real() << ", " << expected[u][v].imag()
                    << "), Got (" << G[u][v].real() << ", " << G[u][v].imag() << ")" << endl;
            }
        }
    }

    cout << "IFFT2D (Reconstructed Image) Result:" << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cout << G[i][j].real() << " "; // Output only the real part
        }
        cout << endl;
    }

    // Cleanup dynamically allocated arrays
    for (int i = 0; i < height; ++i) {
        delete[] F_shifted[i];
        delete[] G[i];
        delete[] expected[i];
    }
    delete[] F_shifted;
    delete[] G;
    delete[] expected;

    return test_passed;
}
