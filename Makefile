CXXFLAGS=-O3 -g -std=c++20
USE_WARP_PRIMITIVES?=yes

ifeq ($(USE_WARP_PRIMITIVES),no)
CXXFLAGS+= -DNO_WARP_PRIMITIVES
endif

ifeq ($(USE_OPENBLAS),no)
CXXFLAGS+= -DNO_OPENBLAS
else
CXXFLAGS+= -lopenblas
endif

# set up thread count for multithreaded CPU matrix multiply
OPENBLAS_NUM_THREADS?="$$(nproc)"

# Default value for Tesla T4. Use TARGET=sm_86 for RTX3060
TARGET?=sm_75

all: a2

a2: main.cu dense_graph.h sparse_graph.h data_types.h data_generator.h cuda_common.h 
	OPENBLAS_NUM_THREADS=$(OPENBLAS_NUM_THREADS) nvcc -o a2 main.cu -arch=$(TARGET) $(CXXFLAGS)

clean:
	rm -f a2
