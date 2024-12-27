CXXFLAGS=-O3 -g -std=c++20

ifeq ($(USE_OPENBLAS),no)
CXXFLAGS+= -DNO_OPENBLAS
else
CXXFLAGS+= -lopenblas
endif

# set up thread count for multithreaded CPU matrix multiply
OPENBLAS_NUM_THREADS?="$$(nproc)"

# Test options
TEST_N?=4096
TEST_MAX_ELEMENT?=1
# Will vary by GPU, so users should specify at the command line
NUM_SMS?=28


# Default value for Turing. Use TARGET=sm_86 for Ampere
TARGET?=sm_75

all: bench

bench: main.cu gemm_experiment.h cuda_common.h GEMM.hpp
	OPENBLAS_NUM_THREADS=$(OPENBLAS_NUM_THREADS) nvcc -o bench main.cu \
    -arch=$(TARGET) -DNUM_SMS=$(NUM_SMS) -DTEST_N=$(TEST_N) \
    -DTEST_MAX_ELEMENT=$(TEST_MAX_ELEMENT) $(CXXFLAGS)

clean:
	rm -f bench
