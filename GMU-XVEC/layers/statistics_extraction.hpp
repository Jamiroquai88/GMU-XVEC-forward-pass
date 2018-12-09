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
    std::vector<float> forward(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context);
    
private:
    bool include_variance;
};


#endif /* statistics_extraction_h */
