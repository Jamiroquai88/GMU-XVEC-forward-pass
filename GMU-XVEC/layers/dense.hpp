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
    DenseLayer(std::string name, cl_context context, std::vector<float> linear, std::vector<float> bias);
    cl_mem forward(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue);
    void Free();
    
private:
    cl_mem m_linear;
    unsigned long m_linear_size;
    cl_mem m_bias;
    cl_mem m_output;
    cl_mem m_input;
};


#endif /* dense_h */
