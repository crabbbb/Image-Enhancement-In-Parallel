#pragma once
#include <cuComplex.h>
#include <iostream>

uint8_t* convertCuComplex2DToUint8(cuDoubleComplex** complexImage, int width, int height);
cuDoubleComplex** convertUint8ToCuComplex2D(const uint8_t* grayscale, int width, int height);
bool testGrayscaleComplexConversion();