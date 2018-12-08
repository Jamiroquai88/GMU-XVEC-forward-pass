//
//  utils.hpp
//  GMU-XVEC
//
//  Created by Jan Profant on 28/11/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#ifndef utils_h
#define utils_h

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iterator>
#include <unordered_map>


std::vector<std::string> split(const std::string &s, char delim);
bool startswith(const std::string& input, const std::string& match);
bool is_in(const std::vector<std::string>& input, const std::string element);
bool is_in(std::unordered_map<std::string, std::string>& input, const std::string element);
std::string join(std::vector<std::string> strings, std::string delim=" ");
std::vector<int> str2ints(std::string str);

void transpose(float *src, float *dst, const unsigned long N, const unsigned long M);

std::vector<float> loadtxt(std::string fea_path, unsigned long &num_samples, unsigned long &num_dims);
void savetxt(std::string output_fname, std::vector<float> matrix, unsigned long cols, unsigned long rows);

bool allclose(std::vector<float> v1, std::vector<float> v2, float eps=0.0001);


#endif /* utils_h */
