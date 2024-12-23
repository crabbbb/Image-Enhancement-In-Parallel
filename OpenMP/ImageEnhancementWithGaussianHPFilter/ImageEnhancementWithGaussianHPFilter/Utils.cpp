#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <opencv2/opencv.hpp>

using namespace std;

bool storeDataIntoFile(double time, string fname) {

    string fileName = fname + ".txt";

    // read file 
    ifstream readFile(fileName, ios::in);

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
            ofstream appendFile(fileName, ios::app);

            appendFile << time << endl;
            appendFile.close();

            return true;
        }
    }

    // if bigger than 10 
    // overwrite the file 
    ofstream writeFile(fileName);

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



