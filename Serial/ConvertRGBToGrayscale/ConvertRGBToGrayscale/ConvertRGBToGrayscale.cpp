// ConvertRGBToGrayscale.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include "convertGrayscale.hpp"
#include "imageUtil.hpp"

using namespace std;

int main()
{
    string imagePath = "C:\\Users\\LENOVO\\OneDrive\\Documents\\GitHub\\Image-Enhancement-In-Parallel\\resource\\raw\\lena.jpeg";

    cv::Mat rgbImage = readImage(imagePath);

    if (rgbImage.empty()) {
        // dont have data 
        printf("Unable to retreive image place check the file path");
        return -1;
    }

    if (rgbImage.channels() < 3) {
        // not RGB
        printf("Image is not in RGB");
        return -1;
    }

    // get width and height 
    int width = rgbImage.cols;
    int height = rgbImage.rows;

    // convert image to grayscale 
    uint8_t* grayscaleImage = rgb_to_grayscale(rgbImage.data, width, height);

    // check unsuccess 
    if (!grayscaleImage) {
        printf("RGB convert Grayscale unsuccess");
        return -1;
    }

    // convert back
    cv::Mat grayscaleMat(height, width, CV_8UC1, grayscaleImage);

    cv::imshow("Original Image", rgbImage);
    cv::imshow("GrayScale Image", grayscaleMat);
    cv::waitKey(0);

    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
