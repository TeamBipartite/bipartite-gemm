#include <chrono>   // for timing
#include <iostream> // std::cout, std::endl
#include <iterator> // std::ostream_iterator
#include <cstdlib> // EXIT_SUCCESS
#include <vector>

#ifndef NO_OPENBLAS
#include <cblas.h>
#endif

#include "utils.h"
#include "GEMM.h"

using namespace csc485b::a4;

/** 
 * main
 * @brief Performs GEMM on GPU CUDA Cores and Tensor Cores.
 */
int main(int argc, char **argv)
{   
    bool print_result = false;

    if (argc > 1 && !strncmp(argv[1], "-p", 3)) print_result = true;

    constexpr int n =  utils::get_padded_sz(64, 16);
    constexpr int max_element = 10;
    
    /*
    *******************************
    * Prepare Matrices
    *******************************
    */
    const std::vector<half> matrix_a = utils::generate_matrix<half>( max_element, n*n );
    half* d_matrix_a;

    const std::vector<half> matrix_b = utils::generate_matrix<half>( max_element, n*n );
    half* d_matrix_b;

    std::vector<half> matrix_c(n*n, 0);
    half* d_matrix_c;

    // Copy data to device
    utils::allocate_device_space( &d_matrix_a, &d_matrix_b, &d_matrix_c, n*n );

    // Copy contents of matrix_a and matrix_b to device
    cudaMemcpy( d_matrix_a, matrix_a.data(), sizeof(half) * matrix_a.size(), cudaMemcpyHostToDevice );
    cudaMemcpy( d_matrix_b, matrix_b.data(), sizeof(half) * matrix_b.size(), cudaMemcpyHostToDevice );

    // Set contents of matrix_c to zero on device
    cudaMemset(d_matrix_c, 0x0, sizeof(half) * matrix_c.size() );

    /*
    *******************************
    * CUDA Core GEMM Implementation
    *******************************
    */

    // Perform GEMM of matrix a x b, storing the result in matrix c
    uint32_t block_dim_sz = (uint32_t)(n / 32);
    /*
    auto const cuda_core_gemm_start = std::chrono::high_resolution_clock::now();
    cudacores::matrix_mult<<< dim3{block_dim_sz, block_dim_sz, block_dim_sz}, dim3{32, 32, 1} >>>(d_matrix_a, d_matrix_b, d_matrix_c, n);
    cudaDeviceSynchronize();
    auto const cuda_core_gemm_end = std::chrono::high_resolution_clock::now();

    // Copy results back to host
    cudaMemcpy( matrix_c.data(), d_matrix_c, sizeof(uint32_t) * matrix_c.size(), cudaMemcpyDeviceToHost );

    // Get results
    std::cout << "------CUDA Core GEMM Implementation------" << std::endl;
    */
    #ifdef NO_OPENBLAS
        std::vector<uint32_t> matrix_c_expected = utils::naive_cpu_matmul( matrix_a, matrix_b, n );
        utils::print_matrix<uint32_t>( matrix_c_expected, n, print_result );
        utils::print_matrix<uint32_t>( matrix_c, n, print_result );
        std::cout << "Correct Output:" << utils::matrices_equal<uint32_t>(matrix_c, matrix_c_expected) << std::endl;
    #else
        const std::vector<half> matrix_c_float( matrix_c.begin(), matrix_c.end() );
        const std::vector<half> matrix_c_expected = utils::cblas_cpu_matmul<half,half>(matrix_a, matrix_b, n );
        utils::print_matrix<half>( matrix_c_expected, n, print_result );
        utils::print_matrix<half>( matrix_c_float, n, print_result );
        //std::cout << "Correct Output:" << utils::matrices_equal<half>(matrix_c_float, matrix_c_expected) << std::endl;
    #endif
    /*
    std::cout << "CUDA Core GEMM Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(cuda_core_gemm_end - cuda_core_gemm_start).count()
              << " us"
              << std::endl;
    */
    // Reset Device memory of matrix C back to zero
    // TODO: may not need this depending on how we approach the tensor core stuff
    cudaMemset(d_matrix_c, 0, sizeof(uint32_t) * matrix_c.size() );
    
    /*
    *********************************
    * Tensor Core GEMM Implementation
    *********************************
    */
    std::cout << "------Tensor Core GEMM Implementation------" << std::endl;
    auto const tensor_core_gemm_start = std::chrono::high_resolution_clock::now();
    tensorcores::half_gemm<<< 1, dim3{128, 4, 1} >>>(d_matrix_a, d_matrix_b, d_matrix_c, n);
    cudaMemcpy( matrix_c.data(), d_matrix_c, sizeof(half) * matrix_c.size(), cudaMemcpyDeviceToHost );
    auto const tensor_core_gemm_end = std::chrono::high_resolution_clock::now();

    // Get results
    
    utils::print_matrix<half>( matrix_c_expected, n, print_result );
    utils::print_matrix<half>( matrix_c, n, print_result );
    std::cout << "Correct Output:" << utils::matrices_equal<half>(matrix_c, matrix_c_expected) << std::endl;
    std::cout << "Tensor Core GEMM Time: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(tensor_core_gemm_end - tensor_core_gemm_start).count()
              << " us"
              << std::endl;
    
    // Cleanup
    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);

    return EXIT_SUCCESS;
}
