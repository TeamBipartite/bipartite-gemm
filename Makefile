CXXFLAGS=-O3 -g -std=c++20

ifeq ($(USE_OPENBLAS),no)
CXXFLAGS+= -DNO_OPENBLAS
else
CXXFLAGS+= -lopenblas
endif

# set up thread count for multithreaded CPU matrix multiply
OPENBLAS_NUM_THREADS?="$$(nproc)"

# Default value for Turing T4. Use TARGET=sm_86 for RTX3060
TARGET?=sm_75

all: a4

a4: main.cu utils.h cuda_common.h GEMM.h
	OPENBLAS_NUM_THREADS=$(OPENBLAS_NUM_THREADS) nvcc -o a4 main.cu -arch=$(TARGET) $(CXXFLAGS)

clean:
	rm -f a4
