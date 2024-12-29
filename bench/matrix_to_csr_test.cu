#include <cstdlib> // EXIT_SUCCESS
#include <vector>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <numeric>

#include "tempNametempName/GEMM.h"

using namespace tempNametempName;

int main(int argc, char **argv)
{   
    bool print_result = false;

    if (argc > 1 && !strncmp(argv[1], "-p", 3)) print_result = true;


    std::cout<< "-------------Prefix-Sum single block-------------" << std::endl;
    uint32_t* d_arr = nullptr;
    uint32_t* d_final_sum = nullptr;
    uint32_t final_sum = 0;

    uint32_t n1 = 2045;
    std::vector<uint32_t> v1;

    for (uint32_t i = 0; i < n1; ++i ) v1.push_back(i);

    std::vector<uint32_t> v1_copy = v1;

    // Allocate space on device
    cudaMalloc( &d_arr, sizeof( uint32_t ) * n1 );
    cudaMalloc( &d_final_sum, sizeof( uint32_t ) );

    // Copy contents to device
    cudaMemcpy( d_arr, v1.data(), sizeof( uint32_t ) * n1, cudaMemcpyHostToDevice );
    cudaMemcpy( d_final_sum, &final_sum, sizeof( uint32_t ), cudaMemcpyHostToDevice );
    
    cudacores::single_block_prefix_sum<<<1, 1024>>>( d_arr, n1, d_final_sum );

    // Get results from device
    cudaMemcpy( v1.data(), d_arr, sizeof(uint32_t) * n1, cudaMemcpyDeviceToHost );
    cudaMemcpy( &final_sum, d_final_sum, sizeof(uint32_t), cudaMemcpyDeviceToHost );

    // Compute expected results
    std::inclusive_scan(v1_copy.begin(), v1_copy.end(), v1_copy.begin());

    // Pruint results
    bool ps_correct = v1_copy == v1;
    bool final_sum_correct = v1_copy[n1-1] == v1[n1-1];
    std::cout << "Prefix-sum correct: " << ps_correct << std::endl;

    std::cout << "Final sum correct: " << final_sum_correct << std::endl;


    return EXIT_SUCCESS;
}
