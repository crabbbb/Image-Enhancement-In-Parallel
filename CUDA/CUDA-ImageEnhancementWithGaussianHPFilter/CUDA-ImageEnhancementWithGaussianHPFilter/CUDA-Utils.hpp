#pragma once
#include <cuComplex.h>
#include <iostream>

uint8_t* convertToGrayscaleFromCuComplex2D(cuDoubleComplex** complexImage, int width, int height);
cuDoubleComplex** convertToCuComplex2D(const uint8_t* grayscale, int width, int height);
bool testGrayscaleComplexConversion();