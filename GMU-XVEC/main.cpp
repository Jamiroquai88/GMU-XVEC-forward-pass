//
//  main.cpp
//  GMU-XVEC
//
//  Created by Jan Profant on 28/11/2018.
//  Copyright © 2018 Jan Profant. All rights reserved.
//

#include <iostream>
#include <getopt.h>
#include <fstream>
#include <sstream>
#include <string>

#include "nnet.hpp"


int main(int argc, char * argv[]) {
    int c;
    std::string features_path("");
    std::string nnet_path("");
    
    while ((c = getopt(argc, argv, "i:n:")) != -1) {
        switch (c) {
            case 'i':
                features_path = optarg;
                break;
            case 'n':
                nnet_path = optarg;
                break;
            default:
                break;
        }
    }
    if (features_path.length() == 0 || nnet_path.length() == 0) {
        std::cerr << "ERROR: Argument '-i' and '-n' must be specified." << std::endl;
    }
    
    NNet nnet = NNet(nnet_path);
    
 
    return 0;
}