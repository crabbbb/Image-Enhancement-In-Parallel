#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;

bool storeDataIntoFile(double time, string fname, string imName) {

    string filePath = "resource/timetaken/" + fname + "_" + imName + ".txt";
    //string filePath = "../../../resource/timetaken/" + fname + "_" + imName + ".txt";

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

// convert uint8_t* grayscale image to a 2D array of complex numbers
complex<double>** storeUint8ToComplex2D(const uint8_t* image, int width, int height) {
    complex<double>** complex_image = new complex<double>*[height];

    for (int i = 0; i < height; ++i) {
        complex_image[i] = new complex<double>[width];
        for (int j = 0; j < width; ++j) {
            complex_image[i][j] = complex<double>(image[i * width + j], 0.0); // Real part is the intensity, imaginary part is 0
        }
    }

    return complex_image;
}

uint8_t* storeComplex2DToUint8(complex<double>** complex_image, int width, int height) {
    // Allocate memory for the output grayscale image
    uint8_t* grayscale_image = new uint8_t[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // Extract the real part of the complex number
            double real_value = complex_image[i][j].real();

            // Clamp the value to the range [0, 255] to fit in uint8_t
            real_value = clamp(real_value, 0.0, 255.0);

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

uint8_t* zeroPad2D(const uint8_t* input, int oldWidth, int oldHeight, int& newWidth, int& newHeight) {
    // 1. Find the next power of two for both dimensions
    newWidth = nextPowerOfTwo(oldWidth);
    newHeight = nextPowerOfTwo(oldHeight);

    // 2. Allocate new array with the padded dimensions
    uint8_t* padded = new uint8_t[newWidth * newHeight];

    // 3. Initialize the entire array to zeros
    memset(padded, 0, newWidth * newHeight * sizeof(uint8_t));

    // 4. Copy original data into the top-left corner
    for (int r = 0; r < oldHeight; ++r) {
        for (int c = 0; c < oldWidth; ++c) {
            padded[r * newWidth + c] = input[r * oldWidth + c];
        }
    }

    return padded;
}

uint8_t* unzeroPad2D(const uint8_t* padded, int newWidth, int newHeight, int oldWidth, int oldHeight) {
    // 1. Allocate new array for the "cropped" region
    uint8_t* cropped = new uint8_t[oldWidth * oldHeight];

    // 2. Copy data from the top-left corner of the padded array
    for (int r = 0; r < oldHeight; ++r) {
        for (int c = 0; c < oldWidth; ++c) {
            cropped[r * oldWidth + c] = padded[r * newWidth + c];
        }
    }

    return cropped;
}



