//
//  relu.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 01/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef relu_h
#define relu_h

#include <iostream>
#include <string>
#include <vector>

#include "../nnet.hpp"


class ReLULayer : virtual public Layer {
public:
    ReLULayer(std::string name) {};
    cl_mem forward(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue);
};


#endif /* relu_h */
