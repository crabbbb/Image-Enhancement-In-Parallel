#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include "convertGrayscale.hpp"
#include "cv_pipe.h"

using namespace std;

cv::Mat& startProcessing(cv::Mat& in_img) {

    // get width and height 
    int width = in_img.cols;
    int height = in_img.rows;

    // convert image to grayscale 
    uint8_t* grayscaleImage = rgb_to_grayscale(in_img.data, width, height);

    // convert back
    cv::Mat out_img(height, width, CV_8UC1, grayscaleImage);

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
        cv::Mat gray_img;
        // Convert color image to grayscale image
        gray_img = color_to_grayscale(gray_img, image);
        cv_imshow(gray_img);
    }

    return finalize_cv_pipe_comm();
}
