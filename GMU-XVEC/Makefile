CP=g++
CPFLAGS=-std=c++11 -pthread -pedantic -Wall
CPP_FILES := $(wildcard *.cpp)
OBJ_FILES := $(CPP_FILES:.cpp=.o)

all: main

debug: CPFLAGS += -DDEBUG -g
debug: main

main: main.o nnet.o opencl_utils.o utils.o layers/stacking.o layers/dense.o layers/relu.o layers/batchnorm.o
	$(CP) $(CPFLAGS) $^ -o $@ -lm -lOpenCL -lrt

%.o:%.cpp
	$(CP) $(CPFLAGS) -c $< -o $@ -lm -lOpenCL -lrt

clean:
	$(RM) main $(OBJ_FILES) *.h.gch

