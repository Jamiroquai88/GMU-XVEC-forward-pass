//
//  batchnorm.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 01/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef batchnorm_h
#define batchnorm_h

#include <iostream>
#include <string>
#include <vector>

#include "../nnet.hpp"


class BatchNormLayer : virtual public Layer {
public:
    BatchNormLayer(std::string name, cl_context context, std::vector<float> mean, std::vector<float> variance, float epsilon);
    cl_mem forward(cl_mem input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context, cl_command_queue queue);
    void Free();
    
private:
    cl_mem m_mean;
    cl_mem m_variance;
    cl_mem m_epsilon;
};


#endif /* batchnorm_h */
