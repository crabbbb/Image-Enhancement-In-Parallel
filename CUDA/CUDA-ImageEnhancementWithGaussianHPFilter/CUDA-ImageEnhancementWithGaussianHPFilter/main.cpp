#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include "convertGrayscale.hpp"
//#include "cv_pipe.h"
#include "GaussianHPFilter.cuh"
#include "FastFourierTransform.cuh"
#include "Utils.hpp"
#include <filesystem>

using namespace std;

const int N = 10;
const double CUTOFF_FREQUENCY = 100;
const double ALPHA = 1.0;

// all the processing done here 
cv::Mat startProcessing(cv::Mat& in_img, string imName) {

    // get width and height 
    int width = in_img.cols;
    int height = in_img.rows;

    // convert image to grayscale 
    uint8_t* grayscaleImage = rgb_to_grayscale(in_img.data, width, height);

    // start time 
    auto start = chrono::high_resolution_clock::now();

    // convert the grayscale image to a 2D complex array
    cuDoubleComplex** complex_image = convertUint8ToCuComplex2D(grayscaleImage, width, height);

    // Zero-pad the image to power-of-two dimensions
    int paddedWidth, paddedHeight, paddedSquareSize;
    cuDoubleComplex** padded_complex_image = zeroPad2D(
        complex_image,  // original data
        width,          // old width
        height,         // old height
        paddedWidth,       // [out] new width
        paddedHeight       // [out] new height
    );

    

    // Perform forward 2D FFT in-place
    cout << "Performing 2D FFT..." << endl;
    cuDoubleComplex** fftResult = FFT2DParallel(padded_complex_image, paddedWidth, paddedHeight);

    // convert the fft complex to uint8_t with stop the timer 
    auto fftConvertStart = chrono::high_resolution_clock::now();

    uint8_t* fftImage = convertCuComplex2DToUint8(fftResult, paddedWidth, paddedHeight);

    auto fftConvertEnd = chrono::high_resolution_clock::now();

    // apply Gaussian High-Pass Filter
    cout << "Applying unsharp masking with Gaussian High-Pass Filter..." << endl;
    cuDoubleComplex** filteredResult = unsharpMaskingFrequencyDomain(fftResult, paddedWidth, paddedHeight, CUTOFF_FREQUENCY, ALPHA);

    // convert the gaussian complex to uint8_t with stop the timer 
    auto gaussianConvertStart = chrono::high_resolution_clock::now();

    uint8_t* gaussianImage = convertCuComplex2DToUint8(filteredResult, paddedWidth, paddedHeight);

    auto gaussianConvertEnd = chrono::high_resolution_clock::now();

    // perform Inverse FFT
    cout << "Performing Inverse FFT..." << endl;
    cuDoubleComplex** reconstructedImage = IFFT2DParallel(filteredResult, paddedWidth, paddedHeight);

    // end time 
    auto end = chrono::high_resolution_clock::now();

    // crop the image back to original size
    cuDoubleComplex** reconstructedImageWithPaddingRemoved = unzeroPad2D(reconstructedImage, paddedWidth, paddedHeight, width, height);

    // convert the ifft result back to uint8_t 
    uint8_t* ifftImage = convertCuComplex2DToUint8(reconstructedImageWithPaddingRemoved, width, height);

    // calculate the different between fft and gaussian start end 
    auto fftDifferent = (chrono::duration_cast<chrono::milliseconds>(fftConvertEnd - fftConvertEnd)).count();
    auto gaussianDifferent = (chrono::duration_cast<chrono::milliseconds>(gaussianConvertEnd - gaussianConvertStart)).count();

    auto duration = (chrono::duration_cast<chrono::milliseconds>(end - start)).count() - fftDifferent - gaussianDifferent;

    cout << "Total duration time used for CUDA is " << duration << "ms " << endl;

    storeDataIntoFile(duration, "cuda", imName);

    // save image 
    // ------------ this path is for using python execute ------------------------------
    cv::imwrite("resource/result/cuda/" + imName + "_gray.jpg", fromUint8ToMat(grayscaleImage, width, height));
    cv::imwrite("resource/result/cuda/" + imName + "_fft.jpg", fromUint8ToMat(fftImage, width, height));
    cv::imwrite("resource/result/cuda/" + imName + "_gaussian.jpg", fromUint8ToMat(gaussianImage, width, height));

    // ------------ this path is for using visual studio ------------------------------
    //cv::imwrite("../../../resource/result/cuda/" + imName + "_gray.jpg", fromUint8ToMat(grayscaleImage, width, height));
    //cv::imwrite("../../../resource/result/cuda/" + imName + "_fft.jpg", fromUint8ToMat(fftImage, width, height));
    //cv::imwrite("../../../resource/result/cuda/" + imName + "_gaussian.jpg", fromUint8ToMat(gaussianImage, width, height));

    // convert back
    cv::Mat out_img = fromUint8ToMat(ifftImage, width, height);
    cv::imwrite("resource/result/cuda/" + imName + "_ifft.jpg", out_img);
    //cv::imwrite("../../../resource/result/cuda/" + imName + "_ifft.jpg", out_img);

    // Cleanup original complexImage since we no longer need it
    cleanup2DArray(complex_image, height);
    cleanup2DArray(fftResult, paddedHeight);
    cleanup2DArray(filteredResult, paddedHeight);
    cleanup2DArray(reconstructedImage, paddedHeight);
    cleanup2DArray(reconstructedImageWithPaddingRemoved, height);

    return out_img;
}

int main(int argc, char* argv[])
{
    string image[] = { "doggo.jpg", "cameragirl.jpeg", "lena.jpeg", "wolf.jpg" };

    string basePath = "resource/raw/";
    //string basePath = "../../../resource/raw/";

    cv::Mat rgbImage;
    cv::Mat out;

    for (string im : image) {

        string imName = filesystem::path(im).stem().string();
        string completePath = basePath + im;

        for (int i = 0; i < N; i++) {
            rgbImage = cv::imread(completePath);
            out = startProcessing(rgbImage, imName);
        }
    }


    //cv::Mat oriResize;
    //cv::resize(rgbImage, oriResize, cv::Size(600, 600));
    //cv::imshow("Original", oriResize);

    //cv::Mat resultResize;
    //cv::resize(out, resultResize, cv::Size(600, 600));
    //cv::imshow("Result", resultResize);
    //cv::waitKey(0);

    return 0;
}