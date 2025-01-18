#pragma once
#include <complex>
#include <opencv2/opencv.hpp>

using namespace std;

bool storeDataIntoFile(double time, string fname, string imName, int max_line_count);

cv::Mat fromUint8ToMat(uint8_t* grayscaleImage, int width, int height);

// Cleanup a 2D array
void cleanup2DArray(complex<double>**& array, int height);

// Convert uint8_t* grayscale image to a 2D array of complex numbers
complex<double>** storeUint8ToComplex2D(const uint8_t* image, int width, int height);

// Convert 2D array of complex numbers to a uint8_t* grayscale image 
uint8_t* storeComplex2DToUint8(complex<double>** complex_image, int width, int height);

void testStoringToAndFromComplex();

int nextPowerOfTwo(int n);

complex<double>** allocate2DArray(int height, int width);

uint8_t* zeroPad2D(const uint8_t* input, int oldWidth, int oldHeight, int& newWidth, int& newHeight);

uint8_t* unzeroPad2D(const uint8_t* padded, int newWidth, int newHeight, int oldWidth, int oldHeight);