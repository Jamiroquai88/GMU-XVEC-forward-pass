CP=g++
CPFLAGS=-std=c++11 -pthread -pedantic -Wall
CPP_FILES := $(wildcard *.cpp)
OBJ_FILES := $(CPP_FILES:.cpp=.o)

# check for MacOS
OS = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
DARWIN = $(strip $(findstring DARWIN, $(OS)))

ifneq ($(DARWIN),)
	# MacOS System
        CFLAGS += -DMAC
        LIBS=-framework OpenCL
else
	# Linux System
	LIBS=-lOpenCL
endif


all: main

debug: CPFLAGS += -DDEBUG -g
debug: main

main: main.o nnet.o opencl_utils.o utils.o layers/stacking.o layers/dense.o layers/relu.o layers/batchnorm.o layers/statistics_extraction.o layers/statistics_pooling.o layers/convolution.o
	$(CP) $(CPFLAGS) $^ -o $@ -lm $(LIBS)

%.o:%.cpp
	$(CP) $(CPFLAGS) -c $< -o $@ -lm $(LIBS)

clean:
	$(RM) main $(OBJ_FILES) *.h.gch layers/*.o

