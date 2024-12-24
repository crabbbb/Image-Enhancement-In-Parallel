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

// all the processing done here 
cv::Mat startProcessing(cv::Mat& in_img, string imName) {

    // get width and height 
    int width = in_img.cols;
    int height = in_img.rows;

    // ensure image dimensions are in powers of 2
    if (!isPowerOfTwo(width) || !isPowerOfTwo(height)) {
        cout << "Image size is not in powers of 2." << endl;
        cout << "Displying back original image..." << endl;
        return in_img;
    }

    // cutoff frequency for the Gaussian High-Pass Filter
    double cutoff_frequency = 128;

    // convert image to grayscale 
    uint8_t* grayscaleImage = rgb_to_grayscale(in_img.data, width, height);

    // start time 
    auto start = chrono::high_resolution_clock::now();

    // convert the grayscale image to a 2D complex array
    complex<double>** complex_image = convertToComplex2D(grayscaleImage, width, height);

    // Perform forward 2D FFT in-place
    cout << "Performing 2D FFT..." << endl;
    
    FFT2D_inplace(complex_image, width, height, +1);

    // convert the fft complex to uint8_t with stop the timer 
    auto fftConvertStart = chrono::high_resolution_clock::now();

    uint8_t* fftImage = convertToGrayscale(complex_image, width, height);

    auto fftConvertEnd = chrono::high_resolution_clock::now();

    // apply Gaussian High-Pass Filter
    cout << "Applying Gaussian High-Pass Filter..." << endl;
    complex<double>** filteredResult = gaussianHighPassFilter(complex_image, width, height, cutoff_frequency);

    // convert the gaussian complex to uint8_t with stop the timer 
    auto gaussianConvertStart = chrono::high_resolution_clock::now();

    uint8_t* gaussianImage = convertToGrayscale(filteredResult, width, height);

    auto gaussianConvertEnd = chrono::high_resolution_clock::now();

    // perform Inverse FFT
    cout << "Performing Inverse FFT..." << endl;
    // Optionally, perform inverse 2D FFT (just pass sign = -1)
    FFT2D_inplace(filteredResult, width, height, -1);

    // end time 
    auto end = chrono::high_resolution_clock::now();

    // convert the ifft result back to uint8_t 
    uint8_t* ifftImage = convertToGrayscale(filteredResult, width, height);

    // calculate the different between fft and gaussian start end 
    auto fftDifferent = (chrono::duration_cast<chrono::milliseconds>(fftConvertEnd - fftConvertEnd)).count();
    auto gaussianDifferent = (chrono::duration_cast<chrono::milliseconds>(gaussianConvertEnd - gaussianConvertStart)).count();

    auto duration = (chrono::duration_cast<chrono::milliseconds>(end - start)).count() - fftDifferent - gaussianDifferent;

    cout << "Total duration time used for OpenMP is " << duration << "ms " << endl;

    storeDataIntoFile(duration, "omp");

    // save image 
    cv::imwrite("../../../resource/result/omp/" + imName + "_gray.jpg", fromUint8ToMat(grayscaleImage, width, height));
    cv::imwrite("../../../resource/result/omp/" + imName + "_fft.jpg", fromUint8ToMat(fftImage, width, height));
    cv::imwrite("../../../resource/result/omp/" + imName + "_gaussian.jpg", fromUint8ToMat(gaussianImage, width, height));

    // convert back
    cv::Mat out_img = fromUint8ToMat(ifftImage, width, height);
    cv::imwrite("../../../resource/result/omp/" + imName + "_ifft.jpg", out_img);

    return out_img;
}

int main(int argc, char* argv[])
{
    string image[] = { "lena.jpeg", "wolf.jpg" };

    string basePath = "../../../resource/raw/";

    string current = image[0];

    string imName = filesystem::path(current).stem().string();
    string completePath = basePath + current;

    cv::Mat rgbImage;
    cv::Mat out;

    for (int i = 0; i < 10; i++) {
        rgbImage = cv::imread(completePath);
        out = startProcessing(rgbImage, imName);
    }

    cv::Mat oriResize;
    cv::resize(rgbImage, oriResize, cv::Size(600, 600));
    cv::imshow("Original", oriResize);

    cv::Mat resultResize;
    cv::resize(out, resultResize, cv::Size(600, 600));
    cv::imshow("Result", resultResize);
    cv::waitKey(0);

    return 0;
}