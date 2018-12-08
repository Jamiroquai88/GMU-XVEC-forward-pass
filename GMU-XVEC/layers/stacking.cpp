//
//  stacking.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 08/12/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <stdio.h>

#include "stacking.hpp"
#include "../utils.hpp"


std::vector<float> StackingLayer::forward(std::vector<float> input, unsigned long &rows, unsigned long &cols, cl_device_id device, cl_context context) {
    int maxval = offsets[std::max_element(offsets.begin(), offsets.end()) - offsets.begin()];
    int minval = offsets[std::min_element(offsets.begin(), offsets.end()) - offsets.begin()];
    unsigned long output_rows = rows + abs(minval) + maxval;
    unsigned long output_cols = offsets.size() * cols;
    std::vector<float> output(output_cols * output_rows, 0.0f);
    unsigned long input_x = 0;
    unsigned long input_y = 0;
    int pos0;
    unsigned long pos1;
    
    for (unsigned int i = 0; i < offsets.size(); i++) {
        pos0 = abs(maxval) - offsets[i];
        pos1 = pos0 + rows;
        input_x = 0;
        for (unsigned long x1 = pos0; x1 < pos1; x1++) {
            input_y = 0;
            for (unsigned long y1 = i * cols; y1 < (i + 1) * cols; y1++) {
//                std::cout << y1 + x1 * output_cols << " " << input_y + input_x * cols << " " << input[input_y + input_x * cols] << std::endl;
                output[y1 + x1 * output_cols] = input[input_y + input_x * cols];
                input_y++;
            }
            input_x++;
            continue;
        }
    }
  
    rows = rows - (abs(minval) + maxval);
    cols = output_cols;
    output.erase(output.begin(), output.begin() + cols * (abs(minval) + maxval));
    output.erase(output.begin() + rows * cols, output.end());
    return output;
}
