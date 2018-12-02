//
//  statistics_extraction.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 01/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef statistics_extraction_h
#define statistics_extraction_h


#include <iostream>
#include <string>
#include <vector>

#include "../nnet.hpp"


class StatisticsExtractionLayer : virtual public Layer {
public:
    StatisticsExtractionLayer(std::string name, bool include_variance) : include_variance(include_variance) {};
    float * forward(float *input, unsigned long num_samples, unsigned long num_dims);
    
private:
    bool include_variance;
};


float * StatisticsExtractionLayer::forward(float *input, unsigned long num_samples, unsigned long num_dims) {
    return NULL;
}


#endif /* statistics_extraction_h */
