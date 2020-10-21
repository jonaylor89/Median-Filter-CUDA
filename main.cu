
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#define THREAD_DIM 16

using namespace cv;
using namespace std;

__global__ void rgbaToGreyscaleGPU(
    uchar4 *rgbaImage, 
    unsigned char *greyImage,
    int rows,
    int cols
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > cols || y > rows)
    {
        return;
    }

    uchar4 rgba = rgbaImage[y * cols + x];
    unsigned char greyValue =  (0.299f * rgba.x) + (0.587f * rgba.y) + (0.114f * rgba.z);
    printf("%c\n", greyValue);
    greyImage[y * cols + x] = greyValue;
}

/*
__global__ void medianFilterGPU(float* greyImageData, int width, int height, float* filteredImage)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int filterWidth = 3;
    int filterHeight = 3;

    int filter[9] {
        0, 1, 0,
        1, 1, 1,
        0, 1, 0
    };

    int pixelValues[9] {0};

    if (x > cols - width + 1 || y > rows - height + 1)
    {
        return;
    }

    int p = 0;
    for (int hh = 0; hh < filterHeight; hh++) 
    {
        for (int ww = 0; ww < filterWidth; ww++) 
        {
            if (filter[hh * filterWidth + ww] == 1)
            {
                int idx = x * width + y + (hh * filterWidth + ww);
                pixel_value[p] = greyImageData[idx];
                p++;
            }
        }
    }

    // Get median pixel value and assign to filteredImage

}
*/

int readImage(
    std::string filename, 
    uchar4 **inputImage, 
    unsigned char **greyImage,
    int *rows,
    int *cols
)
{

    cv::Mat image;
    cv::Mat imageRGBA;
    cv::Mat imageGrey;

    image = cv::imread(filename.c_str(), IMREAD_COLOR);
    if (image.empty())
    {
        cerr << "Couldn't open file: " << filename << endl;
        return 1;
    }

    cv::cvtColor(image, imageRGBA, COLOR_BGR2RGBA);

    imageGrey.create(image.rows, image.cols, CV_8UC1);

    if (!image.isContinuous() || !imageGrey.isContinuous())
    {
        cerr << "Images aren't continous: " << filename << endl;
        return 1;
    }

    *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
    *greyImage = imageGrey.ptr<unsigned char>(0);

    *rows = imageRGBA.rows;
    *cols = imageRGBA.cols;

    return 0;
}

void writeImage(std::string filename, unsigned char *imageGrey)
{
    std::string outFile = "grey_" + filename;
    cv::imwrite(outFile.c_str(), *imageGrey);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        cerr << "Usage: ./main input_file" << endl;
        exit(1);
    }

    // Define Variables
    int err;

    std::string input_file  = std::string(argv[1]);
    
    int rows;
    int cols;
    int size;
    uchar4 *inputImage;
    unsigned char *inputImageGrey;

    uchar4 *d_rgbaImage;
    unsigned char *d_greyImage;

    struct timespec start, end;

    // Read in image
    err = readImage(input_file, &inputImage, &inputImageGrey, &rows, &cols);
    if (err != 0)
    {
        return 1;
    }

    size = rows * cols;

    // Allocate Memory
    cudaMalloc(&d_rgbaImage, sizeof(uchar4) * size);
    cudaMalloc(&d_greyImage, sizeof(unsigned char) * size);

    cudaMemset(&d_greyImage, 0, sizeof(unsigned char) * size);

    // Copy data to GPU
    cudaMemcpy(
        &d_rgbaImage, 
        &inputImage, 
        sizeof(uchar4) * size, 
        cudaMemcpyHostToDevice
    );

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    // Run kernel(s)
    dim3 blockSize (THREAD_DIM, THREAD_DIM);
    dim3 gridSize (ceil(rows / (float)THREAD_DIM), ceil(cols / (float)THREAD_DIM));
    rgbaToGreyscaleGPU<<< gridSize, blockSize >>>(d_rgbaImage, d_greyImage, rows, cols);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("[INFO] Greyscale operation lasted %llu ms\n", diff);

    // Copy results to CPU
    cudaMemcpy(
        &inputImageGrey, 
        &d_greyImage, 
        sizeof(unsigned char) * size, 
        cudaMemcpyDeviceToHost
    );

    // Write Image
    writeImage(input_file, inputImageGrey);

    // Free Memory
    cudaFree(&d_rgbaImage);
    cudaFree(&d_greyImage);

    return 0;
}
