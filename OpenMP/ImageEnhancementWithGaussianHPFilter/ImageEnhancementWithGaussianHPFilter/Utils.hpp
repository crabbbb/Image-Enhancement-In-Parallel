#pragma once
#include <complex>
#include <opencv2/opencv.hpp>

using namespace std;

bool storeDataIntoFile(double time, string fname);

cv::Mat fromUint8ToMat(uint8_t* grayscaleImage, int width, int height);

// Convert uint8_t* grayscale image to a 2D array of complex numbers
complex<double>** convertToComplex2D(const uint8_t* image, int width, int height);

// Convert 2D array of complex numbers to a uint8_t* grayscale image 
uint8_t* convertToGrayscale(complex<double>** complex_image, int width, int height);

bool isPowerOfTwo(int n);