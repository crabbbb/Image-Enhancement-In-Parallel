#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include "convertGrayscale.hpp"
#include "GaussianHPFilter.hpp"
#include "FastFourierTransform.hpp"
#include "InverseFastFourierTransform.hpp"
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

    //uint8_t* grayscaleImage = in_img.data;

    //// convert image to grayscale 
    //if (in_img.channels() == 3) {
    //    grayscaleImage = rgb_to_grayscale(in_img.data, width, height);
    //}

    // convert image to grayscale 
    uint8_t* grayscaleImage = rgb_to_grayscale(in_img.data, width, height);

    // start time 
    auto start = chrono::high_resolution_clock::now();

    // convert the grayscale image to a 2D complex array
    complex<double>** complexImage = convertUint8ToComplex2D(grayscaleImage, width, height);

    // Zero-pad the image to power-of-two dimensions
    int paddedWidth, paddedHeight;
    complex<double>** padded_complex_image = zeroPad2D(
        complexImage,  // original data
        width,          // old width
        height,         // old height
        paddedWidth,       // [out] new width
        paddedHeight       // [out] new height
    );

    // Cleanup original complexImage since we no longer need it
    cleanup2DArray(complexImage, height);

    // perform 2D FFT
    cout << "Performing 2D FFT..." << endl;
    complex<double>** fftResult = FFT2D(padded_complex_image, paddedWidth, paddedHeight);

    // apply Gaussian High-Pass Filter
    cout << "Applying unsharp masking with Gaussian High-Pass Filter..." << endl;
    complex<double>** filteredResult = unsharpMaskingFrequencyDomain(fftResult, paddedWidth, paddedHeight, CUTOFF_FREQUENCY, ALPHA);

    // perform Inverse FFT
    cout << "Performing Inverse FFT..." << endl;
    complex<double>** reconstructedImage = IFFT2D(filteredResult, paddedWidth, paddedHeight);

    // crop the image back to original size
    complex<double>** reconstructedImageWithPaddingRemoved = unzeroPad2D(reconstructedImage, paddedWidth, paddedHeight, width, height);

    // end time 
    auto end = chrono::high_resolution_clock::now();
    auto duration = (chrono::duration_cast<chrono::milliseconds>(end - start)).count();

    cout << "Total duration time used for Serial is " << duration << "ms " << endl;

    storeDataIntoFile(duration, "serial", imName);

    // convert the complex<double> to uint8_t
    uint8_t* frequencyImage = convertComplex2DToUint8(fftResult, paddedWidth, paddedHeight);
    uint8_t* gaussianImage = convertComplex2DToUint8(filteredResult, paddedWidth, paddedHeight);
    uint8_t* spatialImage = convertComplex2DToUint8(reconstructedImageWithPaddingRemoved, width, height);

    // save image 
    // ------------ this path is for using python execute ------------------------------
    cv::imwrite("resource/result/serial/" + imName + "_gray.jpg", fromUint8ToMat(grayscaleImage, width, height));
    cv::imwrite("resource/result/serial/" + imName + "_fft.jpg", fromUint8ToMat(frequencyImage, width, height));
    cv::imwrite("resource/result/serial/" + imName + "_gaussian.jpg", fromUint8ToMat(gaussianImage, width, height));

    // ------------ this path is for using visual studio ------------------------------
    //cv::imwrite("../../../resource/result/serial/" + imName + "_gray.jpg", fromUint8ToMat(grayscaleImage, width, height));
    //cv::imwrite("../../../resource/result/serial/" + imName + "_fft.jpg", fromUint8ToMat(frequencyImage, width, height));
    //cv::imwrite("../../../resource/result/serial/" + imName + "_gaussian.jpg", fromUint8ToMat(gaussianImage, width, height));

    // convert back
    cv::Mat out_img = fromUint8ToMat(spatialImage, width, height);
    cv::imwrite("resource/result/serial/" + imName + "_ifft.jpg", out_img);
    //cv::imwrite("../../../resource/result/serial/" + imName + "_ifft.jpg", out_img);

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
            //for (int i = 0; i < 3; i++) {
            //    out = startProcessing(out, imName);
            //}
        }
    }


    /*cv::Mat oriResize;
    cv::resize(rgbImage, oriResize, cv::Size(600, 600));
    cv::imshow("Original", oriResize);

    cv::Mat resultResize;
    cv::resize(out, resultResize, cv::Size(600, 600));
    cv::imshow("Result", resultResize);
    cv::waitKey(0);*/

    return 0;
}
