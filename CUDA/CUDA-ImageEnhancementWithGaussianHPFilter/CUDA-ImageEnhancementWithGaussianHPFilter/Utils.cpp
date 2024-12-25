#include <iostream>
#include <algorithm>  // for std::min, std::max
#include <cmath>      // for std::round
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;

bool storeDataIntoFile(double time, string fname, string imName) {

    //string filePath = "resource/timetaken/" + fname + "_" + imName + ".txt";
    string filePath = "../../../resource/timetaken/" + fname + "_" + imName + ".txt";

    // read file 
    ifstream readFile(filePath, ios::in);

    int line_count = 0;
    if (readFile.is_open()) {
        string line;

        // read how many line inside 
        while (getline(readFile, line)) {
            line_count++;
        }

        readFile.close();

        if (line_count < 10) {
            // write data into file with append 
            ofstream appendFile(filePath, ios::app);

            appendFile << time << endl;
            appendFile.close();

            return true;
        }
    }

    // if bigger than 10 
    // overwrite the file 
    ofstream writeFile(filePath);

    // check file can be open or not 
    if (writeFile.is_open()) {

        writeFile << time << endl;
        writeFile.close();

        return true;
    }

    return false;
}

cv::Mat fromUint8ToMat(uint8_t* grayscaleImage, int width, int height) {
    cv::Mat out(height, width, CV_8UC1, grayscaleImage);
    return out;
}

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

// Cleanup a 2D array
void cleanup2DArray(cuDoubleComplex**& array, int height) {
    if (array != nullptr) {
        for (int i = 0; i < height; ++i) {
            if (array[i] != nullptr) {
                delete[] array[i];
                array[i] = nullptr; // Prevent dangling pointer
            }
        }
        delete[] array;
        array = nullptr; // Prevent dangling pointer
    }
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

int nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) {
        p <<= 1; // same as p = p * 2
    }
    return p;
}

cuDoubleComplex** allocate2DArray(int height, int width) {
    auto** array2D = new cuDoubleComplex * [height];
    for (int i = 0; i < height; ++i) {
        array2D[i] = new cuDoubleComplex[width];
    }
    return array2D;
}

cuDoubleComplex** zeroPad2D(cuDoubleComplex** input,
    int oldWidth, int oldHeight,
    int& newWidth, int& newHeight)
{
    // 1. Find next power of two in both dimensions
    newWidth = nextPowerOfTwo(oldWidth);
    newHeight = nextPowerOfTwo(oldHeight);


    // 2. Allocate new 2D array with padded dimensions
    cuDoubleComplex** padded = allocate2DArray(newHeight, newWidth);

    // 3. Copy old data to top-left corner and zero-fill the remaining area
    for (int r = 0; r < newHeight; ++r) {
        for (int c = 0; c < newWidth; ++c) {
            if (r < oldHeight && c < oldWidth) {
                // Copy from original
                padded[r][c] = input[r][c];
            }
            else {
                // Outside original region -> zero
                padded[r][c] = make_cuDoubleComplex(0.0, 0.0);
            }
        }
    }

    return padded;
}

//cuDoubleComplex** zeroPad2D(cuDoubleComplex** input,
//    int oldWidth, int oldHeight,
//    int& newWidth, int& newHeight, int& newSize)
//{
//    // 1. Find next power of two in both dimensions
//    newWidth = nextPowerOfTwo(oldWidth);
//    newHeight = nextPowerOfTwo(oldHeight);
//
//    if (newWidth > newHeight) {
//        newSize = newWidth;
//    }
//    else {
//        newSize = newHeight;
//    }
//
//    // 2. Allocate new 2D array with padded dimensions
//    cuDoubleComplex** padded = allocate2DArray(newSize, newSize);
//
//    // 3. Copy old data to top-left corner and zero-fill the remaining area
//    for (int r = 0; r < newSize; ++r) {
//        for (int c = 0; c < newSize; ++c) {
//            if (r < oldHeight && c < oldWidth) {
//                // Copy from original
//                padded[r][c] = input[r][c];
//            }
//            else {
//                // Outside original region -> zero
//                padded[r][c] = make_cuDoubleComplex(0.0, 0.0);
//            }
//        }
//    }
//
//    return padded;
//}

cuDoubleComplex** unzeroPad2D(cuDoubleComplex** padded,
    int newWidth, int newHeight,
    int oldWidth, int oldHeight)
{
    // 1. Allocate new 2D array for the "cropped" region
    cuDoubleComplex** cropped = allocate2DArray(oldHeight, oldWidth);

    // 2. Copy the top-left region from the padded array
    for (int r = 0; r < oldHeight; ++r) {
        for (int c = 0; c < oldWidth; ++c) {
            cropped[r][c] = padded[r][c];
        }
    }

    // 3. Return the "cropped" result
    return cropped;
}