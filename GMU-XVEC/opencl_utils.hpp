//
//  opencl_utils.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 02/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef opencl_utils_h
#define opencl_utils_h

#define CL_SILENCE_DEPRECATION true
#define MAC

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

#endif /* opencl_utils_h */
