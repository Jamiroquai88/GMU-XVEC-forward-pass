//
//  stacking.h
//  GMU-XVEC
//
//  Created by Jan Profant on 01/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef stacking_h
#define stacking_h

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include "../nnet.hpp"


class StackingLayer : virtual public Layer {
public:
    StackingLayer(std::string name, std::vector<int> offsets) : offsets(offsets) {};
    std::vector<float> forward(std::vector<float> input, unsigned long &num_samples, unsigned long &num_dims, cl_device_id device, cl_context context);

private:
    std::vector<int> offsets;
};


std::vector<float> StackingLayer::forward(std::vector<float> input, unsigned long &num_samples, unsigned long &num_dims, cl_device_id device, cl_context context) {
      
    int maxval = offsets[std::max_element(offsets.begin(), offsets.end()) - offsets.begin()];
    int minval = offsets[std::min_element(offsets.begin(), offsets.end()) - offsets.begin()];
    unsigned long cols = num_samples + abs(minval) + maxval;
    unsigned long rows = offsets.size() * num_dims;
    std::vector<float> output(cols * rows, 0.0f);
    
    for (unsigned int i = 0; i < offsets.size(); i++) {
        int pos0 = abs(maxval) - offsets[i];
        unsigned long pos1 = pos0 + num_samples;
        for (unsigned long x1 = i * num_dims; x1 < (i + 1) * num_dims; x1++) {
            for (unsigned long y1 = pos0; y1 < pos1; y1++) {
                output[y1 + x1 * cols] = input[y1 - pos0 + (x1 - i) * num_samples];
            }
            continue;
        }
    }
    unsigned long cutted_cols = num_samples - (maxval + abs(minval));
    unsigned long cutted_rows = rows;
    std::vector<float> cutted_output(cutted_cols * cutted_rows, 0.0f);
    for (unsigned long x = 0; x < cutted_rows; x++)
        for (unsigned long y = 0; y < cutted_cols; y++)
            cutted_output[y + x * cutted_cols] = output[y + maxval + abs(minval) + x * cols];
    
    num_samples = cutted_cols;
    num_dims = cutted_rows;
    return cutted_output;
}


#endif /* stacking_h */
