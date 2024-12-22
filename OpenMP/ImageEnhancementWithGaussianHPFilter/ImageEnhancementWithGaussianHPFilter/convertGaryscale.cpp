#include <opencv2/opencv.hpp>
#include <cstdint>
#include "convertGrayscale.hpp"
#include "imageUtil.hpp"

using namespace std;

uint8_t* rgb_to_grayscale(uint8_t *rgbImage, int width, int height) {
    if (!rgbImage) {
        cout << "Invalid input image pointer!" << endl;
        return nullptr;
    }

    // allocate memory for the grayscale image
    uint8_t* grayscaleImage = new uint8_t[width * height];

    // loop through each pixel
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            // Calculate the pixel index
            int rgbIndex = (row * width + col) * 3;
            int grayscaleIndex = row * width + col;

            // extract B, G, R values
            uint8_t blue = rgbImage[rgbIndex];
            uint8_t green = rgbImage[rgbIndex + 1];
            uint8_t red = rgbImage[rgbIndex + 2];

            // apply the grayscale formula
            grayscaleImage[grayscaleIndex] = static_cast<uint8_t>(0.07 * blue + 0.71 * green + 0.21 * red);
        }
    }

    return grayscaleImage;
}