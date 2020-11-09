
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#define THREAD_DIM 16
#define NUM_STREAMS 16
#define MAX_IMAGE_SIZE (1920 * 1080)

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
    greyImage[y * cols + x] = greyValue;
}

__global__ void medianFilterGPU(unsigned char* greyImageData, unsigned char *filteredImage, int rows, int cols)
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

    if (
        x > cols - windowSize + 1 ||
        y > rows - windowSize + 1 ||
        x < windowSize - 1 ||
        y < windowSize - 1
    )
    {
        return;
    }

    for (int hh = 0; hh < windowSize; hh++) 
    {
        for (int ww = 0; ww < windowSize; ww++) 
        {
            if (filter[hh * windowSize + ww] == 1)
            {
                int idx = (y + hh - 1) * cols + (x + ww - 1);
                pixelValues[hh * windowSize + ww] = greyImageData[idx];
            }
        }
    }

    // Get median pixel value and assign to filteredImage
    for (int i = 0; i < (windowSize * windowSize); i++) {
	    for (int j = i + 1; j < (windowSize * windowSize); j++) {
	        if (pixelValues[i] > pixelValues[j]) {
		        //Swap the variables.
		        char tmp = pixelValues[i];
		        pixelValues[i] = pixelValues[j];
		        pixelValues[j] = tmp;
	        }
	    }
    }

    unsigned char filteredValue = pixelValues[(windowSize * windowSize) / 2];
    filteredImage[y * cols + x] = filteredValue;
}

inline void printTime(string task, struct timespec start, struct timespec end)
{
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    // printf("[INFO] %s operation lasted %llu ms\n", task, diff);
}

void read_directory(string name, vector<string> *v)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        v->push_back(dp->d_name);
    }
    closedir(dirp);
}

void readImage(string filename, Mat* inputImage, Mat* imageGrey)
{

    Mat image;
    Mat imageRGBA;
    Mat outputImage;

    // printf("[DEBUG] %s", "Reading image\n");
    image = imread(filename.c_str(), IMREAD_COLOR);
    if (image.empty())
    {
        cerr << "Couldn't open file: " << filename << endl;
	return;
    }

    // printf("[DEBUG] %s", "Convert color\n");
    cvtColor(image, imageRGBA, COLOR_BGR2RGBA);

    outputImage.create(image.rows, image.cols, CV_8UC1);

    *inputImage = imageRGBA;
    *imageGrey = outputImage;

}

void writeImage(string dirname, string filename, string prefix, Mat outputImage)
{
    string outFile = dirname + string("/") + prefix + filename;

    cv::imwrite(outFile.c_str(), outputImage);
}

int main(int argc, char **argv)
{
    if (argc < 2) 
    {
        cerr << "Usage: ./main inputDirectory" << endl;
        exit(1);
    }

    // Define Variables
    // printf("[DEBUG] operating on directory `%s`\n", argv[1]);
    string inputDir  = string(argv[1]);

    string outputDir = string("motified") + inputDir;

    struct stat st = {0};

    if (stat(outputDir.c_str(), &st) == -1) {
        mkdir(outputDir.c_str(), 0700);
    }

    vector<string> inputFilenames;
    read_directory(inputDir, &inputFilenames);
    
    Mat outputImage;
    vector<Mat> inputImages;
    vector<Mat> outputImages;

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) { cudaStreamCreate(&streams[i]); }

    struct timespec start, end;
    
    uchar4 *d_rgbaImage;
    unsigned char *d_greyImage;
    unsigned char *d_filteredImage;

    // Allocate Memory
    // printf("[DEBUG] %s\n", "Allocating Memory");
    cudaMalloc(&d_rgbaImage, sizeof(uchar4) * MAX_IMAGE_SIZE * NUM_STREAMS);
    cudaMalloc(&d_greyImage, sizeof(unsigned char) * MAX_IMAGE_SIZE * NUM_STREAMS);
    cudaMalloc(&d_filteredImage, sizeof(unsigned char) * MAX_IMAGE_SIZE * NUM_STREAMS);

    // Read in images from the fs
    for (int i = 0; i < inputFilenames.size(); i++)
    {
        Mat imageMat;
        string curImage = inputFilenames[i]; 

        // Read in image
        readImage(
            curImage, 
            &imageMat, 
	    &outputImage
        );

        inputImages.push_back(imageMat);
    }

    for (int i = 0; i < inputImages.size(); i++)
    {


        const int curStream = i % NUM_STREAMS; 
        Mat curImageMat = inputImages[i];

        int rows = curImageMat.rows;
        int cols = curImageMat.cols;
        int size = rows * cols;
 
        cudaMemsetAsync(d_greyImage + MAX_IMAGE_SIZE * curStream, 0, sizeof(unsigned char) * MAX_IMAGE_SIZE, streams[curStream]);
        cudaMemsetAsync(d_filteredImage + MAX_IMAGE_SIZE * curStream, 0, sizeof(unsigned char) * MAX_IMAGE_SIZE, streams[curStream]);

        dim3 gridSize (ceil(cols / (float)THREAD_DIM), ceil(rows / (float)THREAD_DIM));
        dim3 blockSize (THREAD_DIM, THREAD_DIM);

        // Copy data to GPU
        // printf("[DEBUG] %s\n", "Copying memory to GPU");
        cudaMemcpyAsync(
            d_rgbaImage + MAX_IMAGE_SIZE * curStream, 
            (uchar4 *)curImageMat.ptr<unsigned char>(0), 
            sizeof(uchar4) * size, 
            cudaMemcpyHostToDevice,
            streams[curStream]
        );

        // Run kernel(s)
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        rgbaToGreyscaleGPU<<< gridSize, blockSize, 0, streams[curStream] >>>(
            d_rgbaImage + (MAX_IMAGE_SIZE * curStream), d_greyImage + (MAX_IMAGE_SIZE * curStream), rows, cols);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        printTime("Greyscale", start, end);

        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        medianFilterGPU<<< gridSize, blockSize, 0, streams[curStream] >>>(d_greyImage, d_filteredImage, rows, cols);
        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        printTime("Median Filter", start, end);

        // Copy results to CPU
        unsigned char *outputImagePtr = outputImage.ptr<unsigned char>(0);
        cudaMemcpyAsync(
            outputImagePtr,
            d_filteredImage + MAX_IMAGE_SIZE * curStream, 
            sizeof(unsigned char) * size, 
            cudaMemcpyDeviceToHost,
            streams[curStream]
        );
        Mat outputImageMat = Mat(rows, cols, CV_8UC1, outputImagePtr);
        outputImages.push_back(outputImageMat);
    }

    // sync and destroy streams
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Write modified images to the fs
    for (int i = 0; i < outputImages.size(); i++)
    {
        // Write Image
        writeImage(outputDir, to_string(i) + string(".jpg"), "modified_", outputImages[i]);
    }

    // Free Memory
    cudaFree(&d_rgbaImage);
    cudaFree(&d_greyImage);

    return 0;
}
