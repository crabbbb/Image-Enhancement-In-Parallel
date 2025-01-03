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
cv::Mat startProcessing(cv::Mat& in_img, string imName, int cutoff_frequency, double alpha, bool isArgument = true) {

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
    cuDoubleComplex** fftResult = FFT2DParallel(padded_image, paddedWidth, paddedHeight);

    // convert the fft complex to uint8_t with stop the timer 
    auto fftConvertStart = chrono::high_resolution_clock::now();

    uint8_t* fftImage = storeCuComplex2DToUint8(fftResult, paddedWidth, paddedHeight);

    auto fftConvertEnd = chrono::high_resolution_clock::now();

    // apply Gaussian High-Pass Filter
    cout << "Applying unsharp masking with Gaussian High-Pass Filter..." << endl;
    cuDoubleComplex** filteredResult = unsharpMaskingFrequencyDomain(fftResult, paddedWidth, paddedHeight, cutoff_frequency, alpha);

    // convert the gaussian complex to uint8_t with stop the timer 
    auto gaussianConvertStart = chrono::high_resolution_clock::now();

    uint8_t* gaussianImage = storeCuComplex2DToUint8(filteredResult, paddedWidth, paddedHeight);

    auto gaussianConvertEnd = chrono::high_resolution_clock::now();

    // perform Inverse FFT
    cout << "Performing Inverse FFT..." << endl;
    uint8_t* reconstructedImage = IFFT2DParallel(filteredResult, paddedWidth, paddedHeight);

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

    cout << "Total duration time used for CUDA is " << duration << "ms " << endl;

    // only when is not argument then can write the file 
    if (!isArgument) {
        storeDataIntoFile(duration, "cuda", imName);
    }
    
    // save image 
    // ------------ this path is for using python execute ------------------------------
    cv::imwrite("resource/result/cuda/" + imName + "_gray.jpg", fromUint8ToMat(channelImage, width, height));
    cv::imwrite("resource/result/cuda/" + imName + "_fft.jpg", fromUint8ToMat(fftImage, width, height));
    cv::imwrite("resource/result/cuda/" + imName + "_gaussian.jpg", fromUint8ToMat(gaussianImage, width, height));

    // ------------ this path is for using visual studio ------------------------------
    //cv::imwrite("../../../resource/result/cuda/" + imName + "_gray.jpg", fromUint8ToMat(channelImage, width, height));
    //cv::imwrite("../../../resource/result/cuda/" + imName + "_fft.jpg", fromUint8ToMat(fftImage, width, height));
    //cv::imwrite("../../../resource/result/cuda/" + imName + "_gaussian.jpg", fromUint8ToMat(gaussianImage, width, height));

    // convert back
    cv::Mat out_img = fromUint8ToMat(ifftImage, width, height);
    cv::imwrite("resource/result/cuda/" + imName + "_ifft.jpg", out_img);
    //cv::imwrite("../../../resource/result/cuda/" + imName + "_ifft.jpg", out_img);

    // Cleanup original complexImage since we no longer need it
    cleanup2DArray(fftResult, paddedHeight);
    cleanup2DArray(filteredResult, paddedHeight);

    return out_img;
}

bool isImageFile(const string& file_path) {
    // List of valid image extensions
    const vector<string> valid_extensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif" };

    // Get file extension
    string extension = filesystem::path(file_path).extension().string();

    // Convert to lowercase for case-insensitive comparison
    transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    // Check if extension is in the list of valid image extensions
    return find(valid_extensions.begin(), valid_extensions.end(), extension) != valid_extensions.end();
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

        cv::Mat processedChannel = startProcessing(bgrChannels[c], imName, cutoff_frequency, alpha, false);

        cv::imwrite("resource/result/cuda/" + imName + "channel_" + color + ".jpg", processedChannel);

        bgrChannels[c] = processedChannel;
    }

    cv::Mat mergedResult;
    cv::merge(bgrChannels, mergedResult);
    cv::imwrite("resource/result/cuda/" + imName + "merged_result.jpg", mergedResult);
}

void processGreyscale()
{
    string image[] = { "doggo.jpg", "cameragirl.jpeg", "lena.jpeg", "wolf.jpg" };

    string basePath = "resource/raw/";
    // string basePath = "../../../resource/raw/";

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

void processArguments(int argc, char* argv[]) {
    if (argc < 5) {
        cout << "Insufficient number of arguments." << endl;
        cout << "Usage: path/to/exe_file_to_execute -single [cutoff frequency] [alpha] path/to/image_file" << endl;
    }

    int cutoff_frequency;
    double alpha;
    string file_path = argv[4];
    bool valid_cutoff_frequency = false;
    bool valid_alpha = false;
    bool valid_file = false;
    cout << "cutoff_frequency" << "\n" << endl;

    try {
        cutoff_frequency = stoi(argv[2]);
        if (cutoff_frequency < 0 || cutoff_frequency > 255) {
            cout << "Cutoff frequency must be an integer between 0 and 255." << endl;
        }
        else {
            cout << "Cutoff frequency is valid: " << cutoff_frequency << endl;
            valid_cutoff_frequency = true;
        }
    }
    catch (const exception& e) {
        cout << "Invalid cutoff frequency. Please provide an integer." << endl;
    }

    try {
        alpha = stod(argv[3]);
        if (alpha > 0) {
            cout << "Alpha value is valid for sharpening: " << alpha << endl;
            valid_alpha = true;
        }
        else {
            cout << "Alpha value must be greater than 0 for sharpening." << endl;
        }
    }
    catch (const exception& e) {
        cout << "Invalid alpha value. Please provide a valid floating-point number." << endl;
    }

    // File path validation
    if (filesystem::exists(file_path)) {
        if (isImageFile(file_path)) {
            cout << "File path is valid, and it is an image: " << file_path << endl;
            valid_file = true;
        }
        else {
            cout << "Invalid file type. Please provide an image file (e.g. .jpg, .jpeg, .png, .bmp, .tiff, .gif)." << endl;
        }
    }
    else {
        cout << "Invalid file path. The specified file does not exist: " << file_path << endl;
    }

    if (valid_cutoff_frequency && valid_alpha && valid_file) {
        cout << "All inputs are valid. Proceeding with processing..." << endl;
        processRGB(cutoff_frequency, alpha, file_path);
    }
    else {
        cout << "Usage: path/to/exe_file_to_execute -single [cutoff frequency] [alpha] path/to/image_file" << endl;
    }
}

int main(int argc, char* argv[])
{
    if (argc > 1 && strcmp(argv[1], "-single") == 0) {
        processArguments(argc, argv);
        return 0;
    }

    processGreyscale();

    return 0;
}