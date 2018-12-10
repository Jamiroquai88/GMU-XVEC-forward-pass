//
//  main.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 28/11/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <iostream>
#include <getopt.h>
#include <fstream>
#include <sstream>
#include <string>

#include "nnet.hpp"
#include "utils.hpp"
#include "opencl_utils.hpp"
#include "layers/stacking.hpp"
#include "layers/dense.hpp"
#include "layers/relu.hpp"
#include "layers/batchnorm.hpp"
#include "layers/statistics_extraction.hpp"
#include "layers/statistics_pooling.hpp"

#define MAC
#define CL_SILENCE_DEPRECATION true

#ifdef MAC
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif


std::vector<float> test_stacking_layer(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context) {
    std::vector<int> offsets;
    offsets.push_back(-2);
    offsets.push_back(-1);
    offsets.push_back(0);
    offsets.push_back(1);
    offsets.push_back(2);
    StackingLayer layer = StackingLayer("test", offsets);
    std::vector<float> output = layer.forward(input, rows, cols, device, context);
    unsigned long output_rows, output_cols;
    std::vector<float> ref = loadtxt("tests/ref_stacking_layer.txt", output_rows, output_cols);
    if (!allclose(output, ref)) {
        std::cerr << "TEST FAIL: tests/ref_stacking_layer.txt" << std::endl;
        exit(1);
    }
    return output;
}


std::vector<float> test_dense_layer(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context) {
    unsigned long linear_rows, linear_cols;
    std::vector<float> linear = loadtxt("tests/linear_dense_layer.txt", linear_rows, linear_cols);
    std::vector<float> bias = {0.5, 1.5, -1.5, -0.5};
    DenseLayer layer = DenseLayer("", linear, bias);
    std::vector<float> output = layer.forward(input, rows, cols, device, context);
    unsigned long output_rows, output_cols;
    std::vector<float> ref = loadtxt("tests/ref_dense_layer.txt", output_rows, output_cols);
    if (!allclose(output, ref)) {
        std::cerr << "TEST FAIL: tests/ref_dense_layer.txt" << std::endl;
        exit(1);
    }
    return output;
}


std::vector<float> test_relu_layer(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context) {
    ReLULayer layer = ReLULayer("");
    std::vector<float> output = layer.forward(input, rows, cols, device, context);
    std::vector<float> ref = loadtxt("tests/ref_relu_layer.txt", rows, cols);
    if (!allclose(output, ref)) {
        std::cerr << "TEST FAIL: tests/ref_relu_layer.txt" << std::endl;
        exit(1);
    }
    return output;
}


std::vector<float> test_batchnorm_layer(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context) {
    unsigned long mean_rows, mean_cols, variance_rows, variance_cols;
    std::vector<float> mean = loadtxt("tests/mean_batchnorm_layer.txt", mean_rows, mean_cols);
    std::vector<float> variance = loadtxt("tests/variance_batchnorm_layer.txt", variance_rows, variance_cols);
    float epsilon = 0.0001;
    BatchNormLayer layer = BatchNormLayer("", mean, variance, epsilon);
    std::vector<float> output = layer.forward(input, rows, cols, device, context);
    std::vector<float> ref = loadtxt("tests/ref_batchnorm_layer.txt", rows, cols);
    if (!allclose(output, ref)) {
        std::cerr << "TEST FAIL: tests/ref_batchnorm_layer.txt" << std::endl;
        exit(1);
    }
    return output;
}


void test_statistics_extraction_layer(cl_device_id device, cl_context context) {
    unsigned long rows, cols;
    std::vector<float> input = loadtxt("tests/input_statistics_extraction_layer.txt", rows, cols);
    StatisticsExtractionLayer layer = StatisticsExtractionLayer("", true);
    std::vector<float> output = layer.forward(input, rows, cols, device, context);
    std::vector<float> ref = loadtxt("tests/ref_statistics_extraction_layer.txt", rows, cols);
    if (!allclose(output, ref)) {
        std::cerr << "TEST FAIL: tests/ref_statistics_extraction_layer.txt" << std::endl;
        exit(1);
    }
}


void test_statistics_pooling_layer(cl_device_id device, cl_context context) {
    unsigned long rows, cols;
    StatisticsPoolingLayer layer = StatisticsPoolingLayer("", 15, true, 0, 10000, 1e-10);
    std::vector<float> input = loadtxt("tests/input_statistics_pooling_layer.txt", rows, cols);
    std::vector<float> output = layer.forward(input, rows, cols, device, context);
    std::vector<float> ref = loadtxt("tests/ref_statistics_pooling_layer.txt", rows, cols);
    if (!allclose(output, ref)) {
        std::cerr << "TEST FAIL: tests/ref_statistics_pooling_layer.txt" << std::endl;
        exit(1);
    }
}


bool test(cl_device_id device, cl_context context) {
    std::vector<float> input;
    for (int i = -10; i < 20; i++)
        input.push_back(i);
    unsigned long rows = 6;
    unsigned long cols = 5;
    
    input = test_stacking_layer(input, rows, cols, device, context);
    input = test_dense_layer(input, rows, cols, device, context);
    input = test_relu_layer(input, rows, cols, device, context);
    input = test_batchnorm_layer(input, rows, cols, device, context);
    test_statistics_extraction_layer(device, context);
    test_statistics_pooling_layer(device, context);
    
    std::cerr << "All tests successfully passed." << std::endl;
    return true;
}



/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device(bool use_cpu=false) {
    size_t value_size;
    cl_platform_id platform;
    cl_device_id dev;
    int err;
    
    /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    if(err < 0) {
        std::cerr << "ERROR: Couldn't identify a platform." << std::endl;
        exit(1);
    }
    
    /* Access a device */
    if (!use_cpu) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
        if(err == CL_DEVICE_NOT_FOUND) {
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
        }
    }
    else {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if(err < 0) {
        std::cerr << "ERROR: Couldn't access any devices." << std::endl;
        exit(1);
    }
    clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &value_size);
    char *value = (char*) malloc(value_size);
    clGetDeviceInfo(dev, CL_DEVICE_NAME, value_size, value, NULL);
    std::cout << "Using OpenCL Device: " << value << std::endl;
    free(value);
    
    return dev;
}


int main(int argc, char * argv[]) {
    int c;
    std::string features_path("");
    std::string nnet_path("");
    
    while ((c = getopt(argc, argv, "i:n:")) != -1) {
        switch (c) {
            case 'i':
                features_path = optarg;
                break;
            case 'n':
                nnet_path = optarg;
                break;
            default:
                break;
        }
    }
    if (features_path.length() == 0 || nnet_path.length() == 0) {
        std::cerr << "ERROR: Arguments '-i' and '-n' must be specified." << std::endl;
        exit(1);
    }
    
    cl_device_id device = create_device();
    cl_context context;
    cl_int err;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err < 0) {
        std::cerr << "ERROR: Could not create context." << std::endl;
        exit(1);
    }
    
    test(device, context);
    NNet nnet = NNet(nnet_path);
    nnet.forward(features_path, device, context);
 
    clReleaseContext(context);
    clReleaseDevice(device);
    
    return 0;
}
