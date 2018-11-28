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
#include <sstream>
#include <vector>
#include <iterator>


std::vector<std::string> split(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> result;
    while (std::getline(ss, item, delim)) {
        std::cout << item;
        result.push_back(item);
    }
    return result;
}


bool startswith(const std::string& input, const std::string& match)
{
    return input.size() >= match.size() && equal(match.begin(), match.end(), input.begin());
}


std::string join(std::vector<std::string> strings, std::string delim=" ") {
    std::string result(strings[0]);
    for(auto it = strings.begin(); it != strings.end(); ++it) {
        if (std::next(it) != strings.end() && it != strings.begin()) {
            result += *it + delim;
        }
    }
    return result;
}

#endif /* utils_h */
