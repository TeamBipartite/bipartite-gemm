CXXFLAGS=-O3 -g -std=c++20

ifeq ($(USE_OPENBLAS),no)
CXXFLAGS+= -DNO_OPENBLAS
else
CXXFLAGS+= -lopenblas
endif

# set up thread count for multithreaded CPU matrix multiply
OPENBLAS_NUM_THREADS?="$$(nproc)"

# Will vary by GPU, so users should specify at the command line
NUM_SMS?=28

# Default value for Turing. Use TARGET=sm_86 for Ampere
TARGET?=sm_75

all: a4

a4: main.cu gemm_experiment.h cuda_common.h GEMM.hpp
	OPENBLAS_NUM_THREADS=$(OPENBLAS_NUM_THREADS) nvcc -o a4 main.cu -arch=$(TARGET) -DNUM_SMS=$(NUM_SMS) $(CXXFLAGS)

clean:
	rm -f a4
