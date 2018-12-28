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
#include "layers/convolution.hpp"
#include "layers/max_pooling.hpp"


#define CL_SILENCE_DEPRECATION true

#ifdef MAC
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif


cl_mem test_stacking_layer(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue) {
    std::vector<int> offsets;
    offsets.push_back(-2);
    offsets.push_back(-1);
    offsets.push_back(0);
    offsets.push_back(1);
    offsets.push_back(2);
    StackingLayer layer = StackingLayer("test", offsets);
    cl_mem output = layer.forward(input, rows, cols, device, context, queue);
    unsigned long output_rows, output_cols;
    std::vector<float> ref = loadtxt("tests/ref_stacking_layer.txt", output_rows, output_cols);
    if (!allclose(enqueue_buffer(queue, output, rows, cols), ref)) {
        std::cerr << "TEST FAIL: tests/ref_stacking_layer.txt" << std::endl;
        exit(1);
    }
    layer.Free();
    layer.FreeBase();
    return output;
}


cl_mem test_dense_layer(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue) {
    unsigned long linear_rows, linear_cols;
    std::vector<float> linear = loadtxt("tests/linear_dense_layer.txt", linear_rows, linear_cols);
    std::vector<float> bias = {0.5, 1.5, -1.5, -0.5};
    DenseLayer layer = DenseLayer("", context, linear, bias);
    cl_mem output = layer.forward(input, rows, cols, device, context, queue);
    unsigned long output_rows, output_cols;
    std::vector<float> ref = loadtxt("tests/ref_dense_layer.txt", output_rows, output_cols);
    if (!allclose(enqueue_buffer(queue, output, rows, cols), ref)) {
        std::cerr << "TEST FAIL: tests/ref_dense_layer.txt" << std::endl;
        exit(1);
    }
    layer.Free();
    layer.FreeBase();
    return output;
}


cl_mem test_relu_layer(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue) {
    ReLULayer layer = ReLULayer("");
    cl_mem output = layer.forward(input, rows, cols, device, context, queue);
    std::vector<float> ref = loadtxt("tests/ref_relu_layer.txt", rows, cols);
    if (!allclose(enqueue_buffer(queue, output, rows, cols), ref)) {
        std::cerr << "TEST FAIL: tests/ref_relu_layer.txt" << std::endl;
        exit(1);
    }
    layer.FreeBase();
    return output;
}


cl_mem test_batchnorm_layer(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue) {
    unsigned long mean_rows, mean_cols, variance_rows, variance_cols;
    std::vector<float> mean = loadtxt("tests/mean_batchnorm_layer.txt", mean_rows, mean_cols);
    std::vector<float> variance = loadtxt("tests/variance_batchnorm_layer.txt", variance_rows, variance_cols);
    float epsilon = 0.0001;
    BatchNormLayer layer = BatchNormLayer("", context, mean, variance, epsilon);
    cl_mem output = layer.forward(input, rows, cols, device, context, queue);
    std::vector<float> ref = loadtxt("tests/ref_batchnorm_layer.txt", rows, cols);
    if (!allclose(enqueue_buffer(queue, output, rows, cols), ref)) {
        std::cerr << "TEST FAIL: tests/ref_batchnorm_layer.txt" << std::endl;
        exit(1);
    }
    clReleaseMemObject(output);
    layer.FreeBase();
    return output;
}


void test_statistics_extraction_layer(cl_device_id device, cl_context context, cl_command_queue queue) {
    unsigned long rows, cols;
    std::vector<float> input = loadtxt("tests/input_statistics_extraction_layer.txt", rows, cols);
    cl_int err;
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * input.size(), &input[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    StatisticsExtractionLayer layer = StatisticsExtractionLayer("", true);
    cl_mem output = layer.forward(input_buffer, rows, cols, device, context, queue);
    std::vector<float> ref = loadtxt("tests/ref_statistics_extraction_layer.txt", rows, cols);
    if (!allclose(enqueue_buffer(queue, output, rows, cols), ref)) {
        std::cerr << "TEST FAIL: tests/ref_statistics_extraction_layer.txt" << std::endl;
        exit(1);
    }
    clReleaseMemObject(output);
    layer.Free();
    layer.FreeBase();
}


void test_statistics_pooling_layer(cl_device_id device, cl_context context, cl_command_queue queue) {
    unsigned long rows, cols;
    StatisticsPoolingLayer layer = StatisticsPoolingLayer("", 15, true, 0, 10000, 1e-10);
    std::vector<float> input = loadtxt("tests/input_statistics_pooling_layer.txt", rows, cols);
    cl_int err;
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * input.size(), &input[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_mem output = layer.forward(input_buffer, rows, cols, device, context, queue);
    std::vector<float> ref = loadtxt("tests/ref_statistics_pooling_layer.txt", rows, cols);
    
    if (!allclose(enqueue_buffer(queue, output, rows, cols), ref)) {
        std::cerr << "TEST FAIL: tests/ref_statistics_pooling_layer.txt" << std::endl;
        exit(1);
    }
    layer.Free();
    layer.FreeBase();
}


void test_convolutional_layer(cl_device_id device, cl_context context, cl_command_queue queue) {
    unsigned long rows, cols;
    std::vector<float> input_image = loadtxt("tests/input_convolution_layer.txt", rows, cols);
    std::vector<float> weights = loadtxt("tests/weights_convolution_layer.txt", rows, cols);
    std::vector<float> ref = loadtxt("tests/ref_convolution_layer.txt", rows, cols);
    ConvolutionalLayer layer = ConvolutionalLayer("", context, 7, 7, 3, 2, 3, 3, 2, weights);
    cl_int err;
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * input_image.size(), &input_image[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_mem output = layer.forward(input_buffer, device, context, queue);
    if (!allclose(enqueue_buffer(queue, output, 3, 6), ref)) {
        std::cerr << "TEST FAIL: tests/ref_convolution_layer.txt" << std::endl;
        exit(1);
    }
    layer.Free();
    layer.FreeBase();
    clReleaseMemObject(output);
}


void test_max_pooling_layer(cl_device_id device, cl_context context, cl_command_queue queue) {
    unsigned long rows, cols;
    std::vector<float> input_image = loadtxt("tests/input_max_pooling_layer.txt", rows, cols);
    std::vector<float> ref = loadtxt("tests/ref_max_pooling_layer.txt", rows, cols);
    MaxPoolingLayer layer = MaxPoolingLayer("", 4, 4, 2);
    cl_int err;
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * input_image.size(), &input_image[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_mem output = layer.forward(input_buffer, device, context, queue);
    if (!allclose(enqueue_buffer(queue, output, 4, 2), ref)) {
        std::cerr << "TEST FAIL: tests/ref_max_pooling_layer.txt" << std::endl;
        exit(1);
    }
    layer.Free();
    layer.FreeBase();
    clReleaseMemObject(output);
}



bool test(cl_device_id device, cl_context context, cl_command_queue queue) {
    std::vector<float> input;
    for (int i = -10; i < 20; i++)
        input.push_back(i);
    cl_int err;
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * input.size(), &input[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    unsigned long rows = 6;
    unsigned long cols = 5;
    
    cl_mem output;
    output = test_stacking_layer(input_buffer, rows, cols, device, context, queue);
    output = test_dense_layer(output, rows, cols, device, context, queue);
    output = test_relu_layer(output, rows, cols, device, context, queue);
    output = test_batchnorm_layer(output, rows, cols, device, context, queue);
    test_statistics_extraction_layer(device, context, queue);
    test_statistics_pooling_layer(device, context, queue);
    
    // test convolutional layer and max-pooling
    test_convolutional_layer(device, context, queue);
    test_max_pooling_layer(device, context, queue);
    
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
    std::string output_path("");
    
    while ((c = getopt(argc, argv, "i:n:o:")) != -1) {
        switch (c) {
            case 'i':
                features_path = optarg;
                break;
            case 'n':
                nnet_path = optarg;
                break;
            case 'o':
                output_path = optarg;
                break;
            default:
                break;
        }
    }
    if (features_path.length() == 0 || nnet_path.length() == 0 || output_path.length() == 0) {
        std::cerr << "ERROR: Arguments '-i', '-o' and '-n' must be specified." << std::endl;
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
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    test(device, context, queue);
    NNet nnet = NNet(nnet_path, context);
    unsigned long rows, cols;
    std::vector<float> output = nnet.forward(features_path, device, context, queue, rows, cols);
    savetxt(output_path, output, rows, cols);
 
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);
    
    return 0;
}
