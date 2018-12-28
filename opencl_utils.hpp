//
//  opencl_utils.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 02/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef opencl_utils_h
#define opencl_utils_h

#include <string>
#include <vector>


#define CL_SILENCE_DEPRECATION true

#ifdef MAC
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#pragma comment( lib, "OpenCL" )

const char *getCLError(cl_int err_id);

/* Create program from a file and compile it */
// borrowed from https://github.com/mattscar/opencl_dot_product/blob/master/dot_product.c
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);
cl_kernel compile_kernel(cl_device_id device, cl_context context, std::string src_path, std::string function_name, size_t &max_local_size, cl_program &program);
size_t get_global_group_size(size_t array_size, size_t local_size);
std::vector<float> enqueue_buffer(cl_command_queue queue, cl_mem buffer, unsigned long rows, unsigned long cols);

#endif /* opencl_utils_h */
