#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;

bool storeDataIntoFile(double time, string fname, string imName) {

    //string filePath = "resource/timetaken/" + fname + "_" + imName + ".txt";
    string filePath = "../../../resource/timetaken/" + fname + "_" + imName + ".txt";

    // read file 
    ifstream readFile(filePath, ios::in);

    int line_count = 0;
    if (readFile.is_open()) {
        std::string line;

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

uint8_t* convertComplex2DToUint8(complex<double>** complex_image, int width, int height) {
    // Allocate memory for the output grayscale image
    uint8_t* grayscale_image = new uint8_t[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // Extract the real part of the complex number
            double real_value = complex_image[i][j].real();

            // Clamp the value to the range [0, 255] to fit in uint8_t
            real_value = std::clamp(real_value, 0.0, 255.0);

            // Store the value as uint8_t in the output array
            grayscale_image[i * width + j] = static_cast<uint8_t>(round(real_value));
        }
    }

    return grayscale_image;
}

int nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) {
        p <<= 1; // same as p = p * 2
    }
    return p;
}

complex<double>** allocate2DArray(int height, int width) {
    auto** array2D = new complex<double>*[height];
    for (int i = 0; i < height; ++i) {
        array2D[i] = new complex<double>[width];
    }
    return array2D;
}

// Cleanup a 2D array
void cleanup2DArray(complex<double>**& array, int height) {
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

complex<double>** zeroPad2D(complex<double>** input,
    int oldWidth, int oldHeight,
    int& newWidth, int& newHeight)
{
    // 1. Find next power of two in both dimensions
    newWidth = nextPowerOfTwo(oldWidth);
    newHeight = nextPowerOfTwo(oldHeight);

    // 2. Allocate new 2D array with padded dimensions
    complex<double>** padded = allocate2DArray(newHeight, newWidth);

    // 3. Copy old data to top-left corner and zero-fill the remaining area
    for (int r = 0; r < newHeight; ++r) {
        for (int c = 0; c < newWidth; ++c) {
            if (r < oldHeight && c < oldWidth) {
                // Copy from original
                padded[r][c] = input[r][c];
            }
            else {
                // Outside original region -> zero
                padded[r][c] = complex<double>(0.0, 0.0);
            }
        }
    }

    return padded;
}

complex<double>** unzeroPad2D(complex<double>** padded,
    int newWidth, int newHeight,
    int oldWidth, int oldHeight)
{
    // 1. Allocate new 2D array for the "cropped" region
    complex<double>** cropped = allocate2DArray(oldHeight, oldWidth);

    // 2. Copy the top-left region from the padded array
    for (int r = 0; r < oldHeight; ++r) {
        for (int c = 0; c < oldWidth; ++c) {
            cropped[r][c] = padded[r][c];
        }
    }

    // 3. Return the "cropped" result
    return cropped;
}



