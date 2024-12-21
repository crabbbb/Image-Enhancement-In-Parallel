#pragma once
#include <complex>

using namespace std;

cv::Mat fromUint8ToMat(uint8_t* grayscaleImage, int width, int height);

// Cleanup a 2D array
void cleanup2DArray(complex<double>**& array, int height);

// Convert uint8_t* grayscale image to a 2D array of complex numbers
complex<double>** convertToComplex2D(const uint8_t* image, int width, int height);

// Convert 2D array of complex numbers to a uint8_t* grayscale image 
uint8_t* convertToGrayscale(complex<double>** complex_image, int width, int height);

void testConversionToAndFromComplex();