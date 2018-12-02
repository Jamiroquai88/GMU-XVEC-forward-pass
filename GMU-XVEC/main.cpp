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

#define CL_SILENCE_DEPRECATION true
#define MAC

#ifdef MAC
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif


/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {
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
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if(err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }
    if(err < 0) {
        std::cerr << "ERROR: Couldn't access any devices." << std::endl;
        exit(1);
    }
    
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
        std::cerr << "ERROR: Argument '-i' and '-n' must be specified." << std::endl;
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
    
    NNet nnet = NNet(nnet_path);
    nnet.forward(features_path, device, context);
 
    return 0;
}
