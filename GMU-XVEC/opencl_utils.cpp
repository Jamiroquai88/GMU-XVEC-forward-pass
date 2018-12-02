//
//  opencl_utils.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 02/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>
#include "opencl_utils.hpp"

#define CL_SILENCE_DEPRECATION true
#define MAC

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#pragma comment( lib, "OpenCL" )

const char *getCLError(cl_int err_id) {
    switch (err_id)
    {
        case CL_SUCCESS:
            return "Success!";
        case CL_DEVICE_NOT_FOUND:
            return "Device not found.";
        case CL_DEVICE_NOT_AVAILABLE:
            return "Device not available";
        case CL_COMPILER_NOT_AVAILABLE:
            return "Compiler not available";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "Memory object allocation failure";
        case CL_OUT_OF_RESOURCES:
            return "Out of resources";
        case CL_OUT_OF_HOST_MEMORY:
            return "Out of host memory";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "Profiling information not available";
        case CL_MEM_COPY_OVERLAP:
            return "Memory copy overlap";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "Image format mismatch";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "Image format not supported";
        case CL_BUILD_PROGRAM_FAILURE:
            return "Program build failure";
        case CL_MAP_FAILURE:
            return "Map failure";
        case CL_INVALID_VALUE:
            return "Invalid value";
        case CL_INVALID_DEVICE_TYPE:
            return "Invalid device type";
        case CL_INVALID_PLATFORM:
            return "Invalid platform";
        case CL_INVALID_DEVICE:
            return "Invalid device";
        case CL_INVALID_CONTEXT:
            return "Invalid context";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "Invalid queue properties";
        case CL_INVALID_COMMAND_QUEUE:
            return "Invalid command queue";
        case CL_INVALID_HOST_PTR:
            return "Invalid host pointer";
        case CL_INVALID_MEM_OBJECT:
            return "Invalid memory object";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "Invalid image format descriptor";
        case CL_INVALID_IMAGE_SIZE:
            return "Invalid image size";
        case CL_INVALID_SAMPLER:
            return "Invalid sampler";
        case CL_INVALID_BINARY:
            return "Invalid binary";
        case CL_INVALID_BUILD_OPTIONS:
            return "Invalid build options";
        case CL_INVALID_PROGRAM:
            return "Invalid program";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "Invalid program executable";
        case CL_INVALID_KERNEL_NAME:
            return "Invalid kernel name";
        case CL_INVALID_KERNEL_DEFINITION:
            return "Invalid kernel definition";
        case CL_INVALID_KERNEL:
            return "Invalid kernel";
        case CL_INVALID_ARG_INDEX:
            return "Invalid argument index";
        case CL_INVALID_ARG_VALUE:
            return "Invalid argument value";
        case CL_INVALID_ARG_SIZE:
            return "Invalid argument size";
        case CL_INVALID_KERNEL_ARGS:
            return "Invalid kernel arguments";
        case CL_INVALID_WORK_DIMENSION:
            return "Invalid work dimension";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "Invalid work group size";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "Invalid work item size";
        case CL_INVALID_GLOBAL_OFFSET:
            return "Invalid global offset";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "Invalid event wait list";
        case CL_INVALID_EVENT:
            return "Invalid event";
        case CL_INVALID_OPERATION:
            return "Invalid operation";
        case CL_INVALID_GL_OBJECT:
            return "Invalid OpenGL object";
        case CL_INVALID_BUFFER_SIZE:
            return "Invalid buffer size";
        case CL_INVALID_MIP_LEVEL:
            return "Invalid mip-map level";
        default:
            return "Unknown";
    }
}


/* Create program from a file and compile it */
// borrowed from https://github.com/mattscar/opencl_dot_product/blob/master/dot_product.c
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {
    
    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    int err;
    
    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r");
    if(program_handle == NULL) {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);
    
    /* Create program from file */
    program = clCreateProgramWithSource(ctx, 1,
                                        (const char**)&program_buffer, &program_size, &err);
    if(err < 0) {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);
    
    /* Build program */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {
        
        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }
    return program;
}
