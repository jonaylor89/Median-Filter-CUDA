
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

Mat inputImage;
Mat inputImageGrey;

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
    greyImage[y * cols + x] = greyValue;
}

__global__ void medianFilterGPU(unsigned char* greyImageData, int width, int height, unsigned char *filteredImage)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int windowSize = 3;

    int filter[9] {
        0, 1, 0,
        1, 1, 1,
        0, 1, 0
    };

    unsigned char pixelValues[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    if (x > cols - width + 1 || y > rows - height + 1)
    {
        return;
    }

    int p = 0;
    for (int hh = 0; hh < windowSize; hh++) 
    {
        for (int ww = 0; ww < windowSize; ww++) 
        {
            if (filter[hh * windowSize + ww] == 1)
            {
                int idx = x * width + y + (hh * windowSize + ww);
                pixel_value[p] = greyImageData[idx];
                p++;
            }
        }
    }

    // Get median pixel value and assign to filteredImage
    for (int i = 0; i < (windowSize * windowSize); i++) {
	for (int j = i + 1; j < (windowSize * windowSize); j++) {
	    if (pixelValues[i] > pixelValues[j]) {
		//Swap the variables.
		char tmp = filterVector[i];
		pixelValues[i] = pixelValues[j];
		pixelValues[j] = tmp;
	    }
	}
    }

    filteredImage[row * width + col] = pixelValues[(windowSize * windowSize) / 2];

}

int readImage(
    string filename, 
    // uchar4 **inputImage, 
    // unsigned char **inputImageGrey,
    int *rows,
    int *cols
)
{

    Mat image;
    Mat imageRGBA;
    Mat imageGrey;

    printf("[DEBUG] %s", "Reading image\n");
    image = imread(filename.c_str(), IMREAD_COLOR);
    if (image.empty())
    {
        cerr << "Couldn't open file: " << filename << endl;
        return 1;
    }

    printf("[DEBUG] %s", "Convert color\n");
    cvtColor(image, imageRGBA, COLOR_BGR2RGBA);

    imageGrey.create(image.rows, image.cols, CV_8UC1);

    if (!image.isContinuous() || !imageGrey.isContinuous())
    {
        cerr << "Images aren't continous: " << filename << endl;
        return 1;
    }

    printf("[DEBUG] %s", "Convert to pointers\n");
    // inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
    // inputImageGrey = imageGrey.ptr<unsigned char>(0);
    inputImage = imageRGBA;
    inputImageGrey = imageGrey;

    *rows = imageRGBA.rows;
    *cols = imageRGBA.cols;

    return 0;
}

void writeImage(string filename, Mat imageGrey)
{
    string outFile = "grey_" + filename;

    cv::imwrite(outFile.c_str(), imageGrey);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        cerr << "Usage: ./main input_file" << endl;
        exit(1);
    }

    // Define Variables
    int err;

    printf("[DEBUG] operating on file `%s`\n", argv[1]);
    string input_file  = string(argv[1]);
    
    int rows;
    int cols;
    int size;
    // uchar4 *inputImage;
    // unsigned char *inputImageGrey;

    uchar4 *d_rgbaImage;
    unsigned char *d_greyImage;

    struct timespec start, end;

    // Read in image
    err = readImage(
	input_file, 
	// &inputImage, 
	// &inputImageGrey, 
	&rows, 
	&cols
    );
    if (err != 0)
    {
        return 1;
    }

    size = rows * cols;
    printf("[DEBUG] Size is: %d\n", size);

    // Allocate Memory
    printf("[DEBUG] %s\n", "Allocating Memory");
    cudaMalloc(&d_rgbaImage, sizeof(uchar4) * size);
    cudaMalloc(&d_greyImage, sizeof(unsigned char) * size);

    cudaMemset(d_greyImage, 0, sizeof(unsigned char) * size);

    // Copy data to GPU
    printf("[DEBUG] %s\n", "Copying memory to GPU");
    cudaMemcpy(
        d_rgbaImage, 
        (uchar4 *)inputImage.ptr<unsigned char>(0), 
        sizeof(uchar4) * size, 
        cudaMemcpyHostToDevice
    );

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    // Run kernel(s)
    printf("[DEBUG] %s\n", "Running Kernel");
    dim3 blockSize (THREAD_DIM, THREAD_DIM);
    dim3 gridSize (ceil(cols / (float)THREAD_DIM), ceil(rows / (float)THREAD_DIM));
    rgbaToGreyscaleGPU<<< gridSize, blockSize >>>(d_rgbaImage, d_greyImage, rows, cols);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("[INFO] Greyscale operation lasted %llu ms\n", diff);

    

    // Copy results to CPU
    unsigned char *inputImageGreyPtr = inputImageGrey.ptr<unsigned char>(0);
    cudaMemcpy(
	inputImageGreyPtr,
        d_greyImage, 
        sizeof(unsigned char) * size, 
        cudaMemcpyDeviceToHost
    );

    // Write Image
    Mat outputImage = Mat(rows, cols, CV_8UC1, inputImageGreyPtr);
    writeImage(input_file, outputImage);

    // Free Memory
    cudaFree(&d_rgbaImage);
    cudaFree(&d_greyImage);

    return 0;
}
