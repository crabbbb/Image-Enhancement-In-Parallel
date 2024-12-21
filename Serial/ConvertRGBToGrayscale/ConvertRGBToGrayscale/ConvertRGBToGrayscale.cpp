#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include "convertGrayscale.hpp"
#include "cv_pipe.h"

using namespace std;

cv::Mat fromUint8ToMat(uint8_t* grayscaleImage, int width, int height) {
    cv::Mat out(height, width, CV_8UC1, grayscaleImage);
    return out;
}

// all the processing done here 
cv::Mat startProcessing(cv::Mat& in_img) {

    // get width and height 
    int width = in_img.cols;
    int height = in_img.rows;

    // start time 
    auto start = chrono::high_resolution_clock::now();

    // convert image to grayscale 
    uint8_t* grayscaleImage = rgb_to_grayscale(in_img.data, width, height);

    // uint8_t* frequencyImage 
    // uint8_t* gaussianImage 
    // uint8_t* spatialImage

    // end time 
    auto end = chrono::high_resolution_clock::now();
    auto duration = (chrono::duration_cast<chrono::milliseconds>(end - start)).count();

    // save image 
    //cv::imwrite("serial_gray.jpg", fromUint8ToMat(grayscaleImage, width, height));
    //cv::imwrite("serial_frequency.jpg", fromUint8ToMat(frequencyImage, width, height));
    //cv::imwrite("serial_gaussian.jpg", fromUint8ToMat(spatialImage, width, height));

    // convert back
    cv::Mat out_img(height, width, CV_8UC1, grayscaleImage);
    //cv::Mat out_img(height, width, CV_8UC1, spatialImage);
    //cv::imwrite("spatial_inverse.jpg", fromUint8ToMat(spatialImage, width, height));

    return out_img;
}

int main(int argc, char *argv[])
{
    int c;
    std::vector<char*> img_filenames;
    init_cv_pipe_comm(argc, argv, true);

    reset_getopt();
    while ((c = getopt(argc, argv, "p:")) != -1) {
        switch (c) {
        case 'p':
            // Do nothing because it should be handled by cv_pipe
            break;
        case '?':
            // Abort when encountering an unknown option
            return -1;
        }
    }
    // Get all filenames from the non-option arguments
    for (int index = optind; index < argc; index++)
        img_filenames.push_back(argv[index]);

    for (auto filename : img_filenames) {
        std::cout << filename << std::endl;
        // Load the filename image
        cv::Mat image = cv::imread(filename);
        if (image.empty()) {
            std::cerr << "Unable to load image: " << filename << std::endl;
            return -1;
        }
        // Convert color image to grayscale image
        cv::Mat result = startProcessing(image);
        cv_imshow(result);
    }

    return finalize_cv_pipe_comm();
}
