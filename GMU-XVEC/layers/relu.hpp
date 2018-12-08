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
#include "../opencl_utils.hpp"


class ReLULayer : virtual public Layer {
public:
    ReLULayer(std::string name) {};
    std::vector<float> forward(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context);
    
};


#endif /* relu_h */
