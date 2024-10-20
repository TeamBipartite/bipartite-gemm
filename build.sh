#! /binb/bash

# For RTX 3060
nvcc -o a2 main.cu -O3 -g -std=c++20 -arch=sm_86

# For Tesla T4
#nvcc -o a2 main.cu -O3 -g -std=c++20 -arch=sm_75
