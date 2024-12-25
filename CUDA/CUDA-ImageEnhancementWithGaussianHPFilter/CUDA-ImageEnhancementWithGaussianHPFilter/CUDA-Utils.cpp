#include <iostream>
#include <algorithm>  // for std::min, std::max
#include <cmath>      // for std::round
#include <cuda_runtime.h>
#include <cuComplex.h>

uint8_t* convertCuComplex2DToUint8(cuDoubleComplex** complexImage, int width, int height)
{
    // Allocate a 1D grayscale array of size 'width * height'
    uint8_t* grayscale = new uint8_t[width * height];

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // Extract the real part
            double realVal = cuCreal(complexImage[i][j]);

            // Round to nearest integer
            int intVal = static_cast<int>(std::round(realVal));

            // Clamp the value between 0 and 255
            intVal = std::max(0, std::min(255, intVal));

            // Store as uint8_t
            grayscale[i * width + j] = static_cast<uint8_t>(intVal);
        }
    }
    return grayscale;
}

// Converts a 1D grayscale image of size 'width * height' into 
// a 2D array [height][width] of cuDoubleComplex.
// The real part is the grayscale value, and the imaginary part is 0.
cuDoubleComplex** convertUint8ToCuComplex2D(const uint8_t* grayscale, int width, int height)
{
    // Allocate 'height' pointers, each pointing to an array of 'width'
    cuDoubleComplex** complex2D = new cuDoubleComplex * [height];

    for (int row = 0; row < height; ++row) {
        complex2D[row] = new cuDoubleComplex[width];

        // Fill each column of this row
        for (int col = 0; col < width; ++col) {
            double val = static_cast<double>(grayscale[row * width + col]);
            // Store as (val + 0i)
            complex2D[row][col] = make_cuDoubleComplex(val, 0.0);
        }
    }
    return complex2D;
}

bool testGrayscaleComplexConversion()
{
    // Example 4x4 grayscale
    const int width = 4;
    const int height = 4;
    uint8_t originalGray[width * height] = {
        0,  50, 100, 150,
        55,  60,  65,  70,
        200, 201, 202, 203,
        254, 255, 128,   1
    };

    std::cout << "Original Grayscale (4x4):\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << (int)originalGray[i * width + j] << " ";
        }
        std::cout << "\n";
    }

    // 1) Convert from grayscale -> 2D complex
    cuDoubleComplex** complex2D = convertUint8ToCuComplex2D(originalGray, width, height);

    // 2) Convert from 2D complex -> grayscale
    uint8_t* reconstructedGray = convertCuComplex2DToUint8(complex2D, width, height);

    // 3) Compare reconstructedGray to originalGray
    bool match = true;
    for (int i = 0; i < width * height; ++i) {
        if (reconstructedGray[i] != originalGray[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "\n[PASS] Round-trip grayscale <-> cuDoubleComplex2D conversion.\n";
    }
    else {
        std::cout << "\n[FAIL] Mismatch detected in round-trip conversion.\n";
    }

    // Print the reconstructed grayscale
    std::cout << "\nReconstructed Grayscale (4x4):\n";
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << (int)reconstructedGray[i * width + j] << " ";
        }
        std::cout << "\n";
    }

    // Cleanup the 2D array
    for (int row = 0; row < height; ++row) {
        delete[] complex2D[row];
    }
    delete[] complex2D;

    // Cleanup the 1D array returned by convertToGrayscaleFromCuComplex2D
    delete[] reconstructedGray;

    return match;
}