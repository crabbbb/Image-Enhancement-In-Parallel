#pragma once
#include <cuComplex.h>
#include <opencv2/opencv.hpp>

using namespace std;

bool storeDataIntoFile(double time, string fname, string imName, int lineCount);
cv::Mat fromUint8ToMat(uint8_t* grayscaleImage, int width, int height);
uint8_t* storeCuComplex2DToUint8(cuDoubleComplex** complexImage, int width, int height);
cuDoubleComplex** storeUint8ToCuComplex2D(const uint8_t* grayscale, int width, int height);
bool testGrayscaleComplexConversion();
int nextPowerOfTwo(int n);
cuDoubleComplex** allocate2DArray(int height, int width);
uint8_t* zeroPad2D(const uint8_t* input, int oldWidth, int oldHeight, int& newWidth, int& newHeight);
uint8_t* unzeroPad2D(const uint8_t* padded, int newWidth, int newHeight, int oldWidth, int oldHeight);
// Cleanup a 2D array
void cleanup2DArray(cuDoubleComplex**& array, int height);