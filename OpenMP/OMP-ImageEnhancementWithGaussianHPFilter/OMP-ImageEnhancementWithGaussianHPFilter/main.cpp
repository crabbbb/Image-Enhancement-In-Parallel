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

const int N = 1;
const double CUTOFF_FREQUENCY = 100;
const double ALPHA = 1.0;

// all the processing done here 
cv::Mat startProcessing(cv::Mat& in_img, string imName, int cutoff_frequency, double alpha) {

    // get width and height 
    int width = in_img.cols;
    int height = in_img.rows;

    uint8_t* channelImage = in_img.data;

    // start time 
    auto start = chrono::high_resolution_clock::now();

    // Zero-pad the image to power-of-two dimensions
    int paddedWidth, paddedHeight;
    uint8_t* padded_image = zeroPad2D(
        channelImage,  // original data
        width,          // old width
        height,         // old height
        paddedWidth,       // [out] new width
        paddedHeight       // [out] new height
    );

    // Perform forward 2D FFT in-place
    cout << "Performing 2D FFT..." << endl;
    
    complex<double>** fft_results = FFT2D(padded_image, paddedWidth, paddedHeight);

    // convert the fft complex to uint8_t with stop the timer 
    auto fftConvertStart = chrono::high_resolution_clock::now();

    uint8_t* fftImage = storeComplex2DToUint8(fft_results, paddedWidth, paddedHeight);

    auto fftConvertEnd = chrono::high_resolution_clock::now();

    // apply Gaussian High-Pass Filter
    cout << "Applying unsharp masking with Gaussian High-Pass Filter..." << endl;
    complex<double>** filteredResult = unsharpMaskingFrequencyDomain(fft_results, paddedWidth, paddedHeight, cutoff_frequency, alpha);

    // convert the gaussian complex to uint8_t with stop the timer 
    auto gaussianConvertStart = chrono::high_resolution_clock::now();

    uint8_t* gaussianImage = storeComplex2DToUint8(filteredResult, paddedWidth, paddedHeight);

    auto gaussianConvertEnd = chrono::high_resolution_clock::now();

    // perform Inverse FFT
    cout << "Performing Inverse FFT..." << endl;
    // perform inverse 2D FFT
    uint8_t* reconstructedImage = IFFT2D(filteredResult, paddedWidth, paddedHeight);

    // end time 
    auto end = chrono::high_resolution_clock::now();

    // crop the image back to original size
    uint8_t* reconstructedImageWithPaddingRemoved = unzeroPad2D(reconstructedImage, paddedWidth, paddedHeight, width, height);

    // convert the ifft result back to uint8_t 
    uint8_t* ifftImage = reconstructedImageWithPaddingRemoved;

    // calculate the different between fft and gaussian start end 
    auto fftDifferent = (chrono::duration_cast<chrono::milliseconds>(fftConvertEnd - fftConvertEnd)).count();
    auto gaussianDifferent = (chrono::duration_cast<chrono::milliseconds>(gaussianConvertEnd - gaussianConvertStart)).count();

    auto duration = (chrono::duration_cast<chrono::milliseconds>(end - start)).count() - fftDifferent - gaussianDifferent;

    cout << "Total duration time used for OpenMP is " << duration << "ms " << endl;

    //storeDataIntoFile(duration, "omp", imName);

    // save image 
    // ------------ this path is for using python execute ------------------------------
    //cv::imwrite("resource/result/omp/" + imName + "_gray.jpg", fromUint8ToMat(channelImage, width, height));
    //cv::imwrite("resource/result/omp/" + imName + "_fft.jpg", fromUint8ToMat(fftImage, width, height));
    //cv::imwrite("resource/result/omp/" + imName + "_gaussian.jpg", fromUint8ToMat(gaussianImage, width, height));

    // ------------ this path is for using visual studio ------------------------------
    cv::imwrite("../../../resource/result/omp/" + imName + "_gray.jpg", fromUint8ToMat(channelImage, width, height));
    cv::imwrite("../../../resource/result/omp/" + imName + "_fft.jpg", fromUint8ToMat(fftImage, width, height));
    cv::imwrite("../../../resource/result/omp/" + imName + "_gaussian.jpg", fromUint8ToMat(gaussianImage, width, height));

    // convert back
    cv::Mat out_img = fromUint8ToMat(ifftImage, width, height);
    //cv::imwrite("resource/result/omp/" + imName + "_ifft.jpg", out_img);
    cv::imwrite("../../../resource/result/omp/" + imName + "_ifft.jpg", out_img);

    return out_img;
}

// will process multiple color channel with the passing argument 
void processRGB(int cutoff_frequency, double alpha, string file_path)  
{
    cv::Mat rgbImage;
    cv::Mat out;

    string imName = filesystem::path(file_path).stem().string();

    rgbImage = cv::imread(file_path);

    vector<cv::Mat> bgrChannels;
    cv::split(rgbImage, bgrChannels);

    // Resize the output to hold the same number of channels (usually 3 for BGR)
    vector<cv::Mat> output(bgrChannels.size());

    // Process each channel separately.
    for (int c = 0; c < rgbImage.channels(); c++) {

        string color = "";
        switch (c) {
        case 0:
            color = "blue";
            break;
        case 1:
            color = "green";
            break;
        case 2:
            color = "red";
            break;
        }

        cv::Mat processedChannel = startProcessing(bgrChannels[c], imName, cutoff_frequency, alpha);

        cv::imwrite("../../../resource/result/omp/" + imName + "channel_" + color + ".jpg", processedChannel);

        bgrChannels[c] = processedChannel;
    }

    cv::Mat mergedResult;
    cv::merge(bgrChannels, mergedResult);
    cv::imwrite("../../../resource/result/omp/" + imName + "merged_result.jpg", mergedResult);
}

void processSingleImChannel()
{
    //string image[] = { "doggo.jpg", "cameragirl.jpeg", "lena.jpeg", "wolf.jpg" };
    string image[] = { "cameragirl.jpeg" };

    //string basePath = "resource/raw/";
    string basePath = "../../../resource/raw/";

    cv::Mat rgbImage;
    cv::Mat out;

    for (string im : image) {

        string imName = filesystem::path(im).stem().string();
        string completePath = basePath + im;

        for (int i = 0; i < N; i++) {
            rgbImage = cv::imread(completePath);
            out = startProcessing(rgbImage, imName, CUTOFF_FREQUENCY, ALPHA);
        }
    }

    //cv::Mat oriResize;
    //cv::resize(rgbImage, oriResize, cv::Size(600, 600));
    //cv::imshow("Original", oriResize);

    //cv::Mat resultResize;
    //cv::resize(out, resultResize, cv::Size(600, 600));
    //cv::imshow("Result", resultResize);
    //cv::waitKey(0);
}

int main(int argc, char* argv[])
{
    
    

    

    return 0;
}