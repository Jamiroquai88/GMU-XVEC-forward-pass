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
    BatchNormLayer(std::string name, float *mean, float *variance, float epsilon) : mean(mean), variance(variance), epsilon(epsilon) {};
    float * forward(float *input, unsigned long num_samples, unsigned long num_dims);
    
private:
    float *mean;
    float *variance;
    float epsilon;
};


float * BatchNormLayer::forward(float *input, unsigned long num_samples, unsigned long num_dims) {
    return NULL;
}


#endif /* batchnorm_h */
