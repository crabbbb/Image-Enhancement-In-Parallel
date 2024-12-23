#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;

//template <typename T>
//T clamp(T value, T min_value, T max_value) {
//    if (value < min_value) return min_value;
//    if (value > max_value) return max_value;
//    return value;
//}

bool storeDataIntoFile(double dList[], int size, string fname) {

    // Open the file in default mode (truncation mode)
    std::ofstream outFile(fname + ".txt");

    // Check if the file was opened successfully
    if (!outFile) {
        cout << "Error: Could not create or open the file " << endl;
        return false; // Exit with error
    }

    for (int i = 0; i < size; ++i) {
        outFile << dList[i] << endl;
    }

    // Close the file
    outFile.close();

    std::cout << "File overwritten successfully: " << std::endl;
}

cv::Mat fromUint8ToMat(uint8_t* grayscaleImage, int width, int height) {
    cv::Mat out(height, width, CV_8UC1, grayscaleImage);
    return out;
}

uint8_t* convertToGrayscale(complex<double>** complex_image, int width, int height) {
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

// Check if number is power of 2
bool isPowerOfTwo(int n) {
    return n && !(n & (n - 1));
}



