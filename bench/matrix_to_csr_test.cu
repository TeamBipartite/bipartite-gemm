#include <cstdlib> // EXIT_SUCCESS
#include <vector>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <numeric>

#include "tempNametempName/GEMM.h"

using namespace tempNametempName;

int main( )
{   
    std::cout<< "-------------Prefix-Sum single block-------------" << std::endl;
    uint32_t* d_arr = nullptr;
    uint32_t* d_final_sum = nullptr;
    uint32_t final_sum = 0;

    uint32_t n1 = 2047;
    std::vector<uint32_t> v1;

    for (uint32_t i = 0; i < n1; ++i ) v1.push_back(i);

    std::vector<uint32_t> v1_copy = v1;

    // Allocate space on device
    // Note: we allocate n1 + 1 uint32_t's on the device to handle the exclusive scan with HS's algorithm
    cudaMalloc( &d_arr, sizeof( uint32_t ) * (n1+1) );
    cudaMalloc( &d_final_sum, sizeof( uint32_t ) );

    cudaMemset( d_arr, 0, sizeof(uint32_t) * (n1+1));

    // Copy contents to device
    cudaMemcpy( d_arr+1, v1.data(), sizeof( uint32_t ) * n1, cudaMemcpyHostToDevice );
    cudaMemcpy( d_final_sum, &final_sum, sizeof( uint32_t ), cudaMemcpyHostToDevice );
    
    // Note: d_arr has n1+1 elements with the elements of v1 starting at index 1 (with index zero being 0)
    // This allows us to maintain an exlusive scan at indices 0-(n1-1) and still hold the final sum at index n1 if needed
    // We do this because when constructing the V array, we need to know the total number of non-zero elements.
    cudacores::single_block_prefix_sum<<<1, 1024>>>( d_arr+1, n1+1, d_final_sum );

    // Get results from device
    // Exclusive scan is held at indices 0-n1 
    cudaMemcpy( v1.data(), d_arr, sizeof(uint32_t) * n1, cudaMemcpyDeviceToHost );
    cudaMemcpy( &final_sum, d_final_sum, sizeof(uint32_t), cudaMemcpyDeviceToHost );

    // Compute expected results
    std::vector<uint32_t> result(v1_copy.size());
    std::exclusive_scan(v1_copy.begin(), v1_copy.end(), result.begin(), 0);
    std::inclusive_scan(v1_copy.begin(), v1_copy.end(), v1_copy.begin());

    cudaFree(d_arr);
    cudaFree(d_final_sum);

    // Pruint results
    bool ps_correct = result == v1;
    bool final_sum_correct = v1_copy[n1-1] == final_sum;
    std::cout << "Prefix-sum correct: " << ps_correct << std::endl;
    
    for (int i = 0; i < n1; ++i){
        //std::cout << v1[i] << "   ";
    }

    std::cout << std::endl;

    std::cout << "Final sum correct: " << final_sum_correct << std::endl;
    std::cout << "Final sum: " << final_sum << std::endl;


    std::cout<< "-------------Creating the V array-------------" << std::endl;
    uint32_t n2 = 4;
    std::vector input{5, 0, 0, 0, 
                      0, 8, 0, 0,
                      0, 0, 3, 0,
                      0, 6, 0, 0};
    std::vector positions{1, 0, 0, 0, 
                          0, 1, 0, 0,
                          0, 0, 1, 0,
                          0, 1, 0, 0};
    std::exclusive_scan(positions.begin(), positions.end(), positions.begin(), 0);
    int num_non_zeros = 4;
    std::vector<uint32_t> output(num_non_zeros, 0);

    uint32_t* d_input = nullptr;
    uint32_t* d_positions = nullptr;
    uint32_t* d_output = nullptr;

    // Allocate space on device
    cudaMalloc( &d_input, sizeof( uint32_t ) * input.size() );
    cudaMalloc( &d_positions, sizeof( uint32_t ) * positions.size() );
    cudaMalloc( &d_output, sizeof( uint32_t ) * output.size() );

    // Copy contents to device
    cudaMemcpy( d_input, input.data(), sizeof( uint32_t ) * input.size(), cudaMemcpyHostToDevice );
    cudaMemcpy( d_positions, positions.data(), sizeof( uint32_t ) * input.size(), cudaMemcpyHostToDevice );
    cudaMemcpy( d_output, output.data(), sizeof( uint32_t ) * num_non_zeros, cudaMemcpyHostToDevice );
    
    cudacores::insert_non_zero_elements<<<1, 1024>>>( d_input, d_positions, d_output, input.size() );

    // Get results from device
    cudaMemcpy( output.data(), d_output, sizeof(uint32_t) * num_non_zeros, cudaMemcpyDeviceToHost );

    // Compute expected results
    std::vector<uint32_t> output_expected{5, 8, 3, 6};
    bool v_array_correct = output_expected == output;
    std::cout << "Final V array correct: " <<  v_array_correct << std::endl;

    cudaFree( d_input );
    cudaFree( d_output );
    cudaFree( d_positions );

    std::cout<< "-------------Creating COL_INDEX-------------" << std::endl;
    std::vector<uint32_t> row_index{0, 2, 3, 4, 5};
    std::vector<uint32_t> col_index{0, 0, 0, 0, 0};
    std::vector<uint32_t> input2 {5, 1, 0, 0, 
                                  0, 8, 0, 0,
                                  0, 0, 3, 0,
                                  0, 6, 0, 0};
    uint32_t* d_row_index = nullptr;
    uint32_t* d_col_index = nullptr;
    uint32_t* d_input2 = nullptr;

    cudaMalloc( &d_row_index, sizeof( uint32_t ) * row_index.size() );
    cudaMalloc( &d_col_index, sizeof( uint32_t ) * col_index.size() );
    cudaMalloc( &d_input2, sizeof( uint32_t ) * input2.size() );


    cudaMemcpy( d_row_index, row_index.data(), sizeof( uint32_t ) * row_index.size(), cudaMemcpyHostToDevice );
    cudaMemcpy( d_input2, input2.data(), sizeof( uint32_t ) * input2.size(), cudaMemcpyHostToDevice );

    cudacores::store<uint32_t, uint32_t><<<1, 1024>>>( d_input2, d_row_index, d_col_index, n2 );

    // Get results from device
    cudaMemcpy( col_index.data(), d_col_index, sizeof(uint32_t) * col_index.size(), cudaMemcpyDeviceToHost );

    std::vector<uint32_t> expected_col_index{0, 1, 1, 2, 1};

    bool col_index_correct = expected_col_index == col_index;
    std::cout << "col_index correct: " << col_index_correct << std::endl;

    for (int i = 0; i < col_index.size(); ++i){
        std::cout << col_index[i] << "   ";
    }
    std::cout << std::endl;

    cudaFree( d_row_index );
    cudaFree( d_col_index );
    cudaFree( d_input2 );

    return EXIT_SUCCESS;
}
