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


class DenseLayer : virtual public Layer {
public:
    DenseLayer(std::string name, std::vector<float> linear, std::vector<float> bias) : linear(linear), bias(bias), is_transposed(false) {};
    std::vector<float> forward(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context);
    
private:
    std::vector<float> linear;
    std::vector<float> bias;
    bool is_transposed;
};


#endif /* dense_h */
