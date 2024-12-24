#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include "convertGrayscale.hpp"
#include "cv_pipe.h"
#include "GaussianHPFilter.hpp"
#include "FastFourierTransform.hpp"
#include "InverseFastFourierTransform.hpp"
#include "Utils.hpp"

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
