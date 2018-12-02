//
//  dense.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 01/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef dense_h
#define dense_h

#include <iostream>
#include <string>
#include <vector>

#include "../nnet.hpp"
#include "../opencl_utils.hpp"

#define MAC

#ifdef MAC
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif



class DenseLayer : virtual public Layer {
public:
    DenseLayer(std::string name, std::vector<float> linear, std::vector<float> bias) : linear(linear), bias(bias) {};
    std::vector<float> forward(std::vector<float> input, unsigned long num_samples, unsigned long num_dims, cl_device_id device, cl_context context);
    
private:
    std::vector<float> linear;
    std::vector<float> bias;
};


std::vector<float> DenseLayer::forward(std::vector<float> input, unsigned long num_samples, unsigned long num_dims, cl_device_id device, cl_context context) {
    size_t max_local_size;
    /* Allocate output vector - one element for each work-group */
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_local_size), &max_local_size, NULL);
    // TODO - just dummy
    cl_uint num_groups = (cl_uint)(num_samples / 4) / max_local_size;
    cl_program program = build_program(context, device, "layers/dense.cl");
    
    /* Create a kernel for the multiplication function */
    cl_int err;
    cl_kernel dot_kernel = clCreateKernel(program, "dot_product", &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
//    /* Create buffers */
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              sizeof(float) * input.size(), &input[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_mem linear_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float) * linear.size(), &linear[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_groups * sizeof(float), NULL, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
//    if(err < 0) {
//        perror("Couldn't create a buffer");
//        exit(1);
//    };
//    b_buffer = clCreateBuffer(context,
//                              CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
//                              sizeof(b_vec), b_vec, &err);
//    output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
//                                   num_groups * sizeof(float), NULL, &err);

    
    std::vector<float> output;
    return output;
}

#endif /* dense_h */
