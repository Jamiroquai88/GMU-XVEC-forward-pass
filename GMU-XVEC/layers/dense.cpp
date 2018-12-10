//
//  dense.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 08/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>

#include "dense.hpp"
#include "../utils.hpp"
#include "../opencl_utils.hpp"


std::vector<float> DenseLayer::forward(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context) {
    size_t max_local_size;
    cl_int err;
    cl_program program;
    cl_kernel dot_kernel = compile_kernel(device, context, "layers/dense.cl", "dot_product2", max_local_size, program);
    
    unsigned long linear_rows = cols;
    unsigned long linear_cols = linear.size() / cols;
    // transpose matrix if needed because of dot product
//    if (!is_transposed) {
//        std::vector<float> transposed(linear.size());
//        transpose(&linear[0], &transposed[0], linear_cols, linear_rows);
//        linear = transposed;
//        is_transposed = true;
//    }
    
    unsigned long output_rows = rows;
    unsigned long output_cols = linear_cols;
    unsigned long output_size = output_rows * output_cols;
    std::vector<float> output(output_size, 0.0f);
    
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float) * rows * cols, &input[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_mem linear_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float) * linear.size(), &linear[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_mem bias_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(float) * bias.size(), &bias[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * output_rows * output_cols, &output[0], &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Create arguments for multiplication kernel */
    err = clSetKernelArg(dot_kernel, 0, sizeof(unsigned long), &output_size);
    err |= clSetKernelArg(dot_kernel, 1, sizeof(unsigned long), &rows);
    err |= clSetKernelArg(dot_kernel, 2, sizeof(unsigned long), &cols);
    err |= clSetKernelArg(dot_kernel, 3, sizeof(unsigned long), &linear_cols);
    err |= clSetKernelArg(dot_kernel, 4, sizeof(cl_mem), &input_buffer);
    err |= clSetKernelArg(dot_kernel, 5, sizeof(cl_mem), &linear_buffer);
    err |= clSetKernelArg(dot_kernel, 6, sizeof(cl_mem), &bias_buffer);
    err |= clSetKernelArg(dot_kernel, 7, sizeof(cl_mem), &output_buffer);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Enqueue multiplication kernel */
    cl_event prof_event;
    size_t global_size = get_global_group_size(output_size, max_local_size);
    err = clEnqueueNDRangeKernel(queue, dot_kernel, 1, NULL, &global_size,
                                 &max_local_size, 0, NULL, &prof_event);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    
    /* Read output buffer */
    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(float) * output_size, &output[0], 0, NULL, NULL);
    if (err < 0) {
        std::cerr << getCLError(err) << std::endl;
        exit(1);
    }
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(linear_buffer);
    clReleaseMemObject(bias_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseCommandQueue(queue);
    clReleaseEvent(prof_event);
    
//
//    for (unsigned long x = 0; x < output_rows; x++) {
////        float *p = &input[0] + x * cols;
////        for (int i=0; i < cols; i++)
////            std::cout << p[i] << " ";
////        std::cout << std::endl;
//        cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//                                             sizeof(float) * cols, &input[0] + x * cols, &err);
//        if (err < 0) {
//            std::cerr << getCLError(err) << std::endl;
//            exit(1);
//        }
//        for (unsigned long y = 0; y < output_cols; y++) {
////            float *p = &linear[0] + y * cols;
////            for (int i=0; i < cols; i++)
////                std::cout << p[i] << " ";
////            std::cout << std::endl;
//            cl_mem linear_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
//                                                  sizeof(float) * cols, &linear[0] + y * cols, &err);
//            if (err < 0) {
//                std::cerr << getCLError(err) << std::endl;
//                exit(1);
//            }
//            // TODO num groups or what?
//            cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &err);
//            cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
//            if (err < 0) {
//                std::cerr << getCLError(err) << std::endl;
//                exit(1);
//            }
//
//            /* Create arguments for multiplication kernel */
//            err = clSetKernelArg(dot_kernel, 0, sizeof(cl_mem), &input_buffer);
//            err |= clSetKernelArg(dot_kernel, 1, sizeof(cl_mem), &linear_buffer);
//            err |= clSetKernelArg(dot_kernel, 2, sizeof(cl_mem), &output_buffer);
//            err |= clSetKernelArg(dot_kernel, 3, max_local_size * 4 * sizeof(float), NULL);
//            if (err < 0) {
//                std::cerr << getCLError(err) << std::endl;
//                exit(1);
//            }
//
//            /* Enqueue multiplication kernel */
//            cl_event prof_event;
//            size_t global_size = max_local_size * 4;
//            err = clEnqueueNDRangeKernel(queue, dot_kernel, 1, NULL, &global_size,
//                                         &max_local_size, 0, NULL, &prof_event);
//            if (err < 0) {
//                std::cerr << getCLError(err) << std::endl;
//                exit(1);
//            }
//
//            /* Read output buffer */
//            err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, sizeof(float), &output[0] + (x * output_cols + y), 0, NULL, NULL);
//            if (err < 0) {
//                std::cerr << getCLError(err) << std::endl;
//                exit(1);
//            }
//            output[x * output_cols + y] += bias[y];
////            std::cout << output[x * output_cols + y] << " ";
//
//            // release memory objects
//            clReleaseMemObject(linear_buffer);
//            clReleaseMemObject(output_buffer);
//            clReleaseCommandQueue(queue);
//            clReleaseEvent(prof_event);
//        }
////        std::cout << std::endl;
//        clReleaseMemObject(input_buffer);
//    }
    
    /* Read output buffer */
    //    err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, num_groups * sizeof(float), output_vec, 0, NULL, NULL);
    //    if(err < 0) {
    //        perror("Couldn't create a buffer");
    //        exit(1);
    //    };
    //    b_buffer = clCreateBuffer(context,
    //                              CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
    //                              sizeof(b_vec), b_vec, &err);
    //    output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
    //                                   num_groups * sizeof(float), NULL, &err);
    //
    //
    clReleaseKernel(dot_kernel);
    clReleaseProgram(program);
    rows = output_rows;
    cols = output_cols;
    return output;
}
