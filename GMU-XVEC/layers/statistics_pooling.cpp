//
//  statistics_pooling.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 09/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>
#include <math.h>

#include "statistics_pooling.hpp"


std::vector<float> StatisticsPoolingLayer::forward(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context) {
    for (unsigned long i = 0; i < cols; i++)
        input.insert(input.begin(), 0.0f);
    auto start = left_context == 0 ? input.begin() : input.begin() + left_context * cols;
    auto end = right_context > rows ? input.end() : input.begin() + right_context * cols;
    std::vector<float> tosum(end - start);
    std::copy(start, end, tosum.begin());
    std::vector<float> val(cols);
    for (unsigned long i = 0; i < cols; i++)
        val[i] = tosum[cols * rows + i] - tosum[i];
    float cnt = val[0];
    unsigned long dim = (val.size() - 1) / 2;
    
    std::vector<float> output(dim * 2);
    for (unsigned long i = 0; i < dim; i++)
        output[i] = val[i + 1] / cnt;

    float std_val;
    for (unsigned long i = 0; i < dim; i++) {
        std_val = val[dim + 1 + i] / cnt - output[i] * output[i];
        output[i + dim] = sqrt(std_val > variance_floor ? std_val : variance_floor);
    }
    rows = 1;
    cols = dim * 2;
    return output;
}
