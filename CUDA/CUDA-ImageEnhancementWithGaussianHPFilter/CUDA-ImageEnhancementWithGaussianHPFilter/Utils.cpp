#include <complex>
#include <iostream>
#include <iomanip>

using namespace std;

template <typename T>
T clamp(T value, T min_value, T max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

// convert uint8_t* grayscale image to a 2D array of complex numbers
complex<double>** convertToComplex2D(const uint8_t* image, int width, int height) {
    complex<double>** complex_image = new complex<double>*[height];

    for (int i = 0; i < height; ++i) {
        complex_image[i] = new complex<double>[width];
        for (int j = 0; j < width; ++j) {
            complex_image[i][j] = complex<double>(image[i * width + j], 0.0); // Real part is the intensity, imaginary part is 0
        }
    }

    return complex_image;
}

uint8_t* convertToGrayscale(complex<double>** complex_image, int width, int height) {
    // Allocate memory for the output grayscale image
    uint8_t* grayscale_image = new uint8_t[width * height];

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            // Extract the real part of the complex number
            double real_value = complex_image[i][j].real();

            // Clamp the value to the range [0, 255] to fit in uint8_t
            real_value = clamp(real_value, 0.0, 255.0);

            // Store the value as uint8_t in the output array
            grayscale_image[i * width + j] = static_cast<uint8_t>(round(real_value));
        }
    }

    return grayscale_image;
}



// Cleanup a 2D array
void cleanup2DArray(complex<double>**& array, int height) {
    if (array != nullptr) {
        for (int i = 0; i < height; ++i) {
            if (array[i] != nullptr) {
                delete[] array[i];
                array[i] = nullptr; // Prevent dangling pointer
            }
        }
        delete[] array;
        array = nullptr; // Prevent dangling pointer
    }
}

// Check if number is power of 2
bool isPowerOfTwo(int n) {
    return n && !(n & (n - 1));
}

// testing section
void testConversionToAndFromComplex() {
    const int width = 4;
    const int height = 4;

    // Create a test grayscale image
    uint8_t grayscale_image[width * height] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };

    // Print the original grayscale image
    cout << "Original Grayscale Image:" << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cout << setw(3) << static_cast<int>(grayscale_image[i * width + j]) << " ";
        }
        cout << endl;
    }

    // Convert the grayscale image to a complex 2D array
    complex<double>** complex_image = convertToComplex2D(grayscale_image, width, height);

    // Convert the complex image back to grayscale
    uint8_t* reconstructed_image = convertToGrayscale(complex_image, width, height);

    // Print the reconstructed grayscale image
    cout << "Reconstructed Grayscale Image:" << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cout << setw(3) << static_cast<int>(reconstructed_image[i * width + j]) << " ";
        }
        cout << endl;
    }

    // Check if the original and reconstructed images match
    bool test_passed = true;
    for (int i = 0; i < width * height; ++i) {
        if (grayscale_image[i] != reconstructed_image[i]) {
            test_passed = false;
            break;
        }
    }

    if (test_passed) {
        cout << "Test Passed: The reconstructed image matches the original image!" << endl;
    }
    else {
        cout << "Test Failed: The reconstructed image does not match the original image!" << endl;
    }

    // Cleanup
    cleanup2DArray(complex_image, height);
    delete[] reconstructed_image;
}


