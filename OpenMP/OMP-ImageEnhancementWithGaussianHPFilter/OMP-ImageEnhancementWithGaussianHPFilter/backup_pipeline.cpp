#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include "convertGrayscale.hpp"
//#include "cv_pipe.h"
#include "GaussianHPFilter.hpp"
#include "FastFourierTransform.hpp"
#include "Utils.hpp"
#include <filesystem>

using namespace std;

//// all the processing done here 
//cv::Mat startProcessing(cv::Mat& in_img, string imName) {
//
//    // get width and height 
//    int width = in_img.cols;
//    int height = in_img.rows;
//
//    // ensure image dimensions are in powers of 2
//    if (!isPowerOfTwo(width) || !isPowerOfTwo(height)) {
//        cout << "Image size is not in powers of 2." << endl;
//        cout << "Displying back original image..." << endl;
//        return in_img;
//    }
//
//    // cutoff frequency for the Gaussian High-Pass Filter
//    double cutoff_frequency = 128;
//
//    // convert image to grayscale 
//    uint8_t* grayscaleImage = rgb_to_grayscale(in_img.data, width, height);
//
//    // start time 
//    auto start = chrono::high_resolution_clock::now();
//
//    // convert the grayscale image to a 2D complex array
//    complex<double>** complexImage = convertToComplex2D(grayscaleImage, width, height);
//
//    // perform 2D FFT
//    cout << "Performing 2D FFT..." << endl;
//    complex<double>** fftResult = FFT2D(complexImage, width, height);
//
//    auto fft = chrono::high_resolution_clock::now();
//
//    // apply Gaussian High-Pass Filter
//    cout << "Applying Gaussian High-Pass Filter..." << endl;
//    complex<double>** filteredResult = gaussianHighPassFilter(fftResult, width, height, cutoff_frequency);
//
//    auto go = chrono::high_resolution_clock::now();
//
//    // perform Inverse FFT
//    cout << "Performing Inverse FFT..." << endl;
//    complex<double>** reconstructedImage = IFFT2D(filteredResult, width, height);
//
//    // end time 
//    auto end = chrono::high_resolution_clock::now();
//    auto duration = (chrono::duration_cast<chrono::milliseconds>(end - start)).count();
//
//    cout << "Total duration time used for OpenMP is " << duration << "ms " << endl;
//
//    // convert the complex<double> to uint8_t
//    uint8_t* frequencyImage = convertToGrayscale(fftResult, width, height);
//    uint8_t* gaussianImage = convertToGrayscale(filteredResult, width, height);
//    uint8_t* spatialImage = convertToGrayscale(reconstructedImage, width, height);
//
//    // save image 
//    cv::imwrite(imName + "_gray.jpg", fromUint8ToMat(grayscaleImage, width, height));
//    cv::imwrite(imName + "_frequency.jpg", fromUint8ToMat(frequencyImage, width, height));
//    cv::imwrite(imName + "_gaussian.jpg", fromUint8ToMat(gaussianImage, width, height));
//
//    // convert back
//    cv::Mat out_img = fromUint8ToMat(spatialImage, width, height);
//    cv::imwrite(imName + "_inverse.jpg", out_img);
//
//    return out_img;
//}

//int main(int argc, char* argv[])
//{
//    int c;
//    std::vector<char*> img_filenames;
//    init_cv_pipe_comm(argc, argv, true);
//
//    reset_getopt();
//    while ((c = getopt(argc, argv, "p:")) != -1) {
//        switch (c) {
//        case 'p':
//            // Do nothing because it should be handled by cv_pipe
//            break;
//        case '?':
//            // Abort when encountering an unknown option
//            return -1;
//        }
//    }
//    // Get all filenames from the non-option arguments
//    for (int index = optind; index < argc; index++)
//        img_filenames.push_back(argv[index]);
//
//    for (auto filename : img_filenames) {
//        std::cout << filename << std::endl;
//        // Load the filename image
//        cv::Mat image = cv::imread(filename);
//        if (image.empty()) {
//            std::cerr << "Unable to load image: " << filename << std::endl;
//            return -1;
//        }
//        // Convert color image to grayscale image
//        string imName = filesystem::path(filename).stem().string();
//        cv::Mat result = startProcessing(image, imName);
//        cv_imshow(result);
//    }
//
//    return finalize_cv_pipe_comm();
//}

