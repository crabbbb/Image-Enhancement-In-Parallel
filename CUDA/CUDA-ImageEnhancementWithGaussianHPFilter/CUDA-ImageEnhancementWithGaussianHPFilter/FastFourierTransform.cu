/************************************************************************
 * Manual 2D FFT in CUDA using Cooley-Tukey - Supports Rectangular Images
 * ----------------------------------------------------------------------
 * - Accepts/returns cuDoubleComplex** at the host side
 * - Internally flattens data into a 1D GPU buffer for batched processing
 * - Row transform:  length = width, batchCount = height
 * - Transpose       (height x width) -> (width x height)
 * - Column transform: length = height, batchCount = width
 * - Transpose back
 *
 * For inverse transform, each 1D pass scales by (1 / length), so total
 * scaling is (1 / (width*height)).
 ************************************************************************/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <iostream>
#include <cmath>
#include <cstdlib>   // for std::exit
#include <cstring>   // for std::memcpy
#include "Utils.hpp"

using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

 // ---------------------------------------------------------------------
 // Simple CUDA error-checking macro
 // ---------------------------------------------------------------------
#define CHECK_CUDA_ERR(msg) do {                                 \
    cudaError_t err = cudaGetLastError();                        \
    if (err != cudaSuccess) {                                    \
        std::cerr << msg << " (error " << err << "): "           \
                  << cudaGetErrorString(err) << std::endl;       \
        std::exit(EXIT_FAILURE);                                 \
    }                                                            \
} while(0)

// =====================================================================
// 1) Kernels for batched 1D FFT (Cooley–Tukey) in "row-major" slices
// =====================================================================

// (a) Bit-reversal permutation for each "row" of length = `length`
__global__ void BitReverseRowKernel(
    cuDoubleComplex* d_data,
    int length,       // how many elements per row transform
    int batchCount,   // how many rows/batches
    int log2Len       // log2(length) (number of bits in each element)
)
{
    int batch = blockIdx.y;  // which row/batch
    if (batch >= batchCount) return;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        // bit-reverse 'tid' in the range 0 to length - 1
        int x = tid;
        int r = 0;
        // process log2Len bits, after the loop, reversed bits of the current index is assigned to r.
        for (int i = 0; i < log2Len; i++) {
            r = (r << 1) | (x & 1);
            x >>= 1;
        }
        if (r > tid) {
            int rowOffset = batch * length;
            // swap
            cuDoubleComplex tmp = d_data[rowOffset + tid];
            d_data[rowOffset + tid] = d_data[rowOffset + r];
            d_data[rowOffset + r] = tmp;
        }
    }
}

// (b) One pass of the Cooley–Tukey butterfly for each row
__global__ void CooleyTukeyRowKernel(
    cuDoubleComplex* d_data,
    int length,
    int batchCount,
    int halfSize,
    int sign // -1 => forward, +1 => inverse
)
{
    int batch = blockIdx.y;
    if (batch >= batchCount) return;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int totalButterflies = length / 2; // # of butterfly pairs per row

    if (tid < totalButterflies) {
        int blockIndex = tid / halfSize;
        int t = tid % halfSize;
        int butterflySz = 2 * halfSize;

        int rowOffset = batch * length;
        int evenIndex = rowOffset + blockIndex * butterflySz + t;
        int oddIndex = evenIndex + halfSize;

        cuDoubleComplex even = d_data[evenIndex];
        cuDoubleComplex odd = d_data[oddIndex];

        double angle = sign * 2.0 * M_PI * t / (double)butterflySz;
        cuDoubleComplex w = make_cuDoubleComplex(cos(angle), sin(angle));

        cuDoubleComplex product = cuCmul(w, odd);
        d_data[evenIndex] = cuCadd(even, product);
        d_data[oddIndex] = cuCsub(even, product);
    }
}

// (c) Scale kernel: multiply each element in row by (1 / length) if doing inverse
__global__ void ScaleRowKernel(
    cuDoubleComplex* d_data,
    int length,
    int batchCount,
    double scale
)
{
    int batch = blockIdx.y;
    if (batch >= batchCount) return;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length) {
        int idx = batch * length + tid;
        double re = cuCreal(d_data[idx]);
        double im = cuCimag(d_data[idx]);
        d_data[idx] = make_cuDoubleComplex(scale * re, scale * im);
    }
}

// =====================================================================
// 2) Kernel to transpose a (height x width) buffer into (width x height)
//    d_out[col * height + row] = d_in[row * width + col]
// =====================================================================
__global__ void TransposeKernel(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_in,
    int width,
    int height
)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int inIdx = row * width + col;
        int outIdx = col * height + row;
        d_out[outIdx] = d_in[inIdx];
    }
}

// =====================================================================
// 3) Host function to do a batched 1D FFT for either row or column pass
//    length  = number of samples in one row transform
//    batches = how many rows
//    forward = true => sign=-1, else sign=+1 and scale by 1/length
// =====================================================================
static void BatchFFT_1D(
    cuDoubleComplex* d_data,
    int length,
    int batches,
    bool forward
)
{
    // 1) bit-reversal
    // calculates the number of bits needed for the bit-reversal based on the length
    // mathematical equivalent = ceil(log2(length))
    int log2Len = 0;
    while ((1 << log2Len) < length) log2Len++;

    dim3 block(256, 1);
    // ensure enough blocks to cover the length, batches is to run all the row processing in parallel
    dim3 grid((length + block.x - 1) / block.x, batches);

    BitReverseRowKernel << <grid, block >> > (d_data, length, batches, log2Len);
    CHECK_CUDA_ERR("BitReverseRowKernel");

    // 2) iterative passes
    int sign = (forward ? -1 : +1);

    for (int halfSize = 1; halfSize < length; halfSize *= 2) {
        int totalButterflies = length / 2;
        // ensure enough blocks to cover totalButterflies
        dim3 gridBfly((totalButterflies + block.x - 1) / block.x, batches);
        CooleyTukeyRowKernel << <gridBfly, block >> > (d_data, length, batches, halfSize, sign);
        CHECK_CUDA_ERR("CooleyTukeyRowKernel");
    }

    // 3) scale if inverse
    if (!forward) {
        // ensure enough blocks to cover the length
        dim3 gridScale((length + block.x - 1) / block.x, batches);
        double scaleFactor = 1.0 / (double)length;
        ScaleRowKernel << <gridScale, block >> > (d_data, length, batches, scaleFactor);
        CHECK_CUDA_ERR("ScaleRowKernel");
    }
}

// =====================================================================
// 4) Full 2D FFT (rectangular: width x height) in place on d_data
//    forward=true => forward FFT
//    forward=false => inverse FFT
//
//    Step-by-step:
//      - Row pass:   length=width,  batches=height
//      - Transpose:  out = size(width*height)
//      - Column pass: length=height, batches=width
//      - Transpose:  back
//
//    If inverse, each 1D pass scales by (1/length), so total = 1/(width*height).
// =====================================================================
static void FFT2D_CUDA(
    cuDoubleComplex* d_data,
    int width,
    int height,
    bool forward
)
{
    // 1) Row-wise transform: length=width, batches=height
    BatchFFT_1D(d_data, width, height, forward);

    // 2) Transpose (height x width) -> (width x height)
    cuDoubleComplex* d_tmp = nullptr;
    cudaMalloc(&d_tmp, width * height * sizeof(cuDoubleComplex));

    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    TransposeKernel << <grid, block >> > (d_tmp, d_data, width, height);
    CHECK_CUDA_ERR("TransposeKernel #1");

    // 3) Now d_tmp has shape (width x height) in row-major
    //    So "rows" = width, "row length" = height for the next pass
    BatchFFT_1D(d_tmp, height, width, forward);

    // 4) Transpose back (width x height) -> (height x width)
    //    which is effectively your original row-major shape
    //    i.e., d_data again
    //    We just swap the width/height in the transpose call
    {
        dim3 grid2(
            (height + block.x - 1) / block.x,
            (width + block.y - 1) / block.y
        );
        TransposeKernel << <grid2, block >> > (d_data, d_tmp, height, width);
        CHECK_CUDA_ERR("TransposeKernel #2");
    }

    cudaFree(d_tmp);
}

// =====================================================================
// 5) Flatten / unflatten host arrays
// =====================================================================
static void flattenHostArray(
    cuDoubleComplex** input2D,
    cuDoubleComplex* output1D,
    int width,
    int height
)
{
    for (int r = 0; r < height; r++) {
        std::memcpy(
            &output1D[r * width],
            input2D[r],
            width * sizeof(cuDoubleComplex)
        );
    }
}

static void unflattenHostArray(
    const cuDoubleComplex* input1D,
    cuDoubleComplex** output2D,
    int width,
    int height
)
{
    for (int r = 0; r < height; r++) {
        std::memcpy(
            output2D[r],
            &input1D[r * width],
            width * sizeof(cuDoubleComplex)
        );
    }
}

// =====================================================================
// 6) Public function: Forward 2D FFT (rectangular)
// =====================================================================
cuDoubleComplex** FFT2DParallel(uint8_t* inputImage, int width, int height)
{
    cuDoubleComplex** complex_image = storeUint8ToCuComplex2D(inputImage, width, height);

    // Allocate a new host 2D array for the result
    cuDoubleComplex** fft_result = new cuDoubleComplex * [height];
    for (int i = 0; i < height; i++) {
        fft_result[i] = new cuDoubleComplex[width];
    }

    // Flatten input
    cuDoubleComplex* h_temp = new cuDoubleComplex[width * height];
    flattenHostArray(complex_image, h_temp, width, height);

    // GPU alloc
    cuDoubleComplex* d_data = nullptr;
    cudaMalloc(&d_data, width * height * sizeof(cuDoubleComplex));

    // Copy to device
    cudaMemcpy(d_data, h_temp, width * height * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Forward 2D FFT
    FFT2D_CUDA(d_data, width, height, true /* forward */);

    // Copy back
    cudaMemcpy(h_temp, d_data, width * height * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // Unflatten
    unflattenHostArray(h_temp, fft_result, width, height);

    // Cleanup
    delete[] h_temp;
    cudaFree(d_data);

    return fft_result;
}

// =====================================================================
// 7) Public function: Inverse 2D FFT (rectangular)
// =====================================================================
uint8_t* IFFT2DParallel(cuDoubleComplex** freqData, int width, int height)
{
    cuDoubleComplex** spatial_result = new cuDoubleComplex * [height];
    for (int i = 0; i < height; i++) {
        spatial_result[i] = new cuDoubleComplex[width];
    }

    // Flatten
    cuDoubleComplex* h_temp = new cuDoubleComplex[width * height];
    flattenHostArray(freqData, h_temp, width, height);

    // GPU alloc
    cuDoubleComplex* d_data = nullptr;
    cudaMalloc(&d_data, width * height * sizeof(cuDoubleComplex));
    cudaMemcpy(d_data, h_temp, width * height * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Inverse 2D FFT
    FFT2D_CUDA(d_data, width, height, false /* inverse */);

    // Copy back
    cudaMemcpy(h_temp, d_data, width * height * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // Unflatten
    unflattenHostArray(h_temp, spatial_result, width, height);

    // Cleanup
    delete[] h_temp;
    cudaFree(d_data);

    return storeCuComplex2DToUint8(spatial_result, width, height);
}

// =====================================================================
// 8) Example test
//    We'll test with a non-square array, e.g. width=6, height=4
// =====================================================================
//static void printComplex2D(
//    cuDoubleComplex** arr,
//    int w, int h,
//    const char* msg,
//    bool showImag = false
//)
//{
//    std::cout << msg << ":\n";
//    for (int r = 0; r < h; r++) {
//        for (int c = 0; c < w; c++) {
//            double re = cuCreal(arr[r][c]);
//            double im = cuCimag(arr[r][c]);
//            if (!showImag) {
//                // Print real part only (to see if we got back original data)
//                std::cout << re << " ";
//            }
//            else {
//                // Full complex
//                std::cout << "(" << re << "," << im << ") ";
//            }
//        }
//        std::cout << "\n";
//    }
//    std::cout << std::endl;
//}
//
//int main()
//{
//    const int width = 4;
//    const int height = 4;
//
//    uint8_t image[width * height] = {
//        1,2,3,4,
//        5,6,7,8,
//        9,10,11,12,
//        13,14,15,16,
//    };
//
//    // Output initial image
//    cout << "Initial Greyscale Image:" << endl;
//    for (int i = 0; i < height; ++i) {
//        for (int j = 0; j < width; ++j) {
//            cout << static_cast<int>(image[i * width + j]) << " ";
//        }
//        cout << endl;
//    }
//
//    // 1) Forward FFT
//    cuDoubleComplex** freq = FFT2DParallel(image, width, height);
//    printComplex2D(freq, width, height, "Frequency domain (full complex)", true);
//
//    // 2) Inverse FFT
//    uint8_t* recon = IFFT2DParallel(freq, width, height);
//    cout << "Reconstructed (real parts)" << endl;
//    for (int i = 0; i < height; ++i) {
//        for (int j = 0; j < width; ++j) {
//            cout << static_cast<int>(recon[i * width + j]) << " ";
//        }
//        cout << endl;
//    }
//
//    // Cleanup
//    for (int r = 0; r < height; r++) {
//        delete[] freq[r];
//    }
//    delete[] freq;
//
//    return 0;
//}
