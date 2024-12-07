#include <vector>
#include <iostream>
#include <cassert>
#include <random>

#ifndef NO_OPENBLAS
#include <cblas.h>
#endif

namespace csc485b {
namespace a4 {
namespace utils {

/** 
 * get_padded_sz
 * @brief Pads the given n to the least multiple of 32 not less than n
 */
constexpr std::size_t get_padded_sz( std::size_t n )
{
    return n%32 ? n + (32 - n%32) : n;
}

/** 
 * get_random_number
 * @brief Fill given matrix with random numbers in interval [0, upper_bound]
 */
template<typename T>
std::vector<T> generate_matrix( uint32_t upper_bound, std::size_t size )
 {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> distribution(0, upper_bound);

    std::vector<T> matrix;
    for ( std::size_t idx = 0; idx < size; ++idx ){
        matrix.push_back( distribution(rng) );
    }

    return matrix;
 }

/** 
 * allocate_device_space
 * @brief Allocate size amount of space for each pointer on GPU
 */
template<typename T>
void allocate_device_space( T** d_matrix_a, T** d_matrix_b, T** d_matrix_c, std::size_t size){
    cudaMalloc( d_matrix_a, sizeof(T) * size );
    cudaMalloc( d_matrix_b, sizeof(T) * size );
    cudaMalloc( d_matrix_c, sizeof(T) * size );
 }

/** 
 * matrices_equal
 * @brief Returns true if matrix_a and matrix_b are equal (element-wise).
 * Otherwise, returns false.
 */
template< typename T >
bool matrices_equal( const std::vector<T>& matrix_a, const std::vector<T>& matrix_b )
{
    return matrix_a == matrix_b;
}

/** 
 * print_matrix
 * @brief Prints given matrix if enabled is true
 * @pre The given vector is of length nxn
 */
template< typename T >
void print_matrix( const std::vector<T>& matrix, std::size_t n, bool enabled = false )
{
    assert( n*n == matrix.size() && "matrix must be of length n*n");

    if (enabled)
    {
        for (std::size_t idx = 0; idx < n; ++idx)
        {
            std::cout << idx << ": ";
            for (std::size_t jdx = 0; jdx < n; ++jdx)
            {
                // Up-convert to float in case there is no method to print the
                // given type
                std::cout << static_cast<float>(matrix[idx*n + jdx]) << " ";
            }
            std::cout << std::endl;
        }
    }
}

#ifdef NO_OPENBLAS

/** 
 * naive_cpu_matmul
 * @brief Multiplies matrix_a by matrix_b and returns the result as a vector
 *        using naive O(n^3) algorithm
 * @pre matrix_a and matrix_b must be size n*n
 */
template< typename T >
std::vector<T> naive_cpu_matmul(const std::vector<T>& matrix_a, const std::vector<T>& matrix_b, std::size_t n)
{
    // Create a zero-initialized result matrix of size nxn.
    std::vector<T> result( n*n, T(0) );

    for (std::size_t idx = 0; idx < n; idx++)
        for (std::size_t jdx = 0; jdx < n; jdx++)
            for (std::size_t kdx = 0; kdx < n; kdx++)
               result[idx*n + jdx] += matrix_a[idx*n + kdx] * matrix_b[kdx*n + jdx];
    
    return result;
}

#else

/** 
 * cblas_cpu_matmul
 * @brief Multiplies matrix_a by matrix_b and returns the result as a vector
 *        using cblas implementation
 * @pre matrix_a and matrix_b must be size n*n
 */
template<typename I, typename O>
std::vector<O> cblas_cpu_matmul(const std::vector<I>& matrix_a, const std::vector<I>& matrix_b, std::size_t n)
{   
    // Generate vectors of floats to be compatible with cblas
    const std::vector<float> matrix_a_float( matrix_a.begin(), matrix_a.end() );
    const std::vector<float> matrix_b_float( matrix_b.begin(), matrix_b.end() );

    // Create a zero-initialized result matrix of size nxn.
    std::vector<float> result( n*n, 0 );
    cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                 matrix_a_float.data(), n, matrix_b_float.data(), n, 1.0, result.data(), n );
    std::vector<O> result_converted( result.begin(), result.end() ); 
    return result_converted;
}

#endif // NO_OPENBLAS

} // namespace utils
} // namespace a4
} // namepace csc485b
