#! /bin/bash

# set up thread count for multithreaded CPU matrix multiply
#export OPENBLAS_NUM_THREADS=$(nproc)

# For RTX 3060
nvcc -o a2 main.cu -O3 -g -std=c++20 -arch=sm_86 #-lopenblas

# For Tesla T4
#nvcc -o a2 main.cu -O3 -g -std=c++20 -arch=sm_75
