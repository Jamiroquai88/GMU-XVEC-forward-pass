//
//  statistics_extraction.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 09/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>
#include <numeric>
#include <vector>
#include <iostream>
#include <iterator>
#include <functional>

#include "statistics_extraction.hpp"


std::vector<float> StatisticsExtractionLayer::forward(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context) {
    unsigned long output_size = include_variance ? 1 + 2 * cols * rows : 1 + cols;
    std::vector<float> output(output_size, 1.0f);
    std::copy(input.begin(), input.end(), output.begin() + 1);
    
    if (include_variance)
        std::transform(input.begin(), input.end(), input.begin(), output.begin() + 1 + cols, std::multiplies<float>());
    
    cols = output_size;
    return output;
}
