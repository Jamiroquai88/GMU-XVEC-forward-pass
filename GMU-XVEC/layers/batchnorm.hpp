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
    BatchNormLayer(std::string name, std::vector<float> mean, std::vector<float> variance, float epsilon) : mean(mean), variance(variance), epsilon(epsilon) {};
    std::vector<float> forward(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context);
    
private:
    std::vector<float> mean;
    std::vector<float> variance;
    float epsilon;
};


#endif /* batchnorm_h */
