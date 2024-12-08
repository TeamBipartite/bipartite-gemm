#include <vector>
#include <random>
#include <cassert>
#include <chrono>
#include <string>
#include <functional>
#include <mma.h>

#ifndef NO_OPENBLAS
#include <cblas.h>
#endif

namespace csc485b {
namespace a4 {

template < typename I, typename R >
class GemmExperiment {

public:

    // Member Variables
    const std::size_t n;

    const std::vector<I> matrix_a;
    I* d_matrix_a;

    const std::vector<I> matrix_b;
    I* d_matrix_b;

    std::vector<R> matrix_c;
    R* d_matrix_c;

    bool print_result;


    // Member Functions
    GemmExperiment( std::size_t input_n, std::size_t upper_bound, std::size_t multiple, uint32_t seed, bool print_result ):
                                                        n{get_padded_sz(input_n, multiple)},
                                                        matrix_a{generate_matrix<I>(I(upper_bound), input_n, n, seed)}, 
                                                        matrix_b{generate_matrix<I>(I(upper_bound), input_n, n, seed)},
                                                        matrix_c{std::vector<R>(n*n, R(0))},
                                                        d_matrix_a{nullptr},
                                                        d_matrix_b{nullptr},
                                                        d_matrix_c{nullptr},
                                                        fixed_seed{seed},
                                                        print_result{print_result} {}
    
    ~GemmExperiment(){
        if (d_matrix_a != nullptr) cudaFree(&d_matrix_a);
        if (d_matrix_b != nullptr) cudaFree(&d_matrix_b);
        if (d_matrix_c != nullptr) cudaFree(&d_matrix_c);
    }

    void prepare_device(){
        assert( n * n == matrix_a.size() && "GemmExperiment need to be of size n x n" );
        // Allocate space on device
        cudaMalloc( &d_matrix_a, sizeof( I ) * n * n );
        cudaMalloc( &d_matrix_b, sizeof( I ) * n * n );
        cudaMalloc( &d_matrix_c, sizeof( R ) * n * n );

        // Copy contents of matrix_a and matrix_b to device
        cudaMemcpy( d_matrix_a, matrix_a.data(), sizeof( I ) * matrix_a.size(), cudaMemcpyHostToDevice );
        cudaMemcpy( d_matrix_b, matrix_b.data(), sizeof( I ) * matrix_b.size(), cudaMemcpyHostToDevice );

        // Set contents of matrix_c to zero on device
        cudaMemset(d_matrix_c, 0x0, sizeof(R) * matrix_c.size() );
    }

    void get_product_from_device(){
        cudaMemcpy( matrix_c.data(), d_matrix_c, sizeof(R) * matrix_c.size(), cudaMemcpyDeviceToHost );
    }

    std::size_t get_n(){
        return n;
    }

    void run_experiment( std::function<void(I*, I*, R*)> kernel_wrapper, std::string title){
        prepare_device();
        auto const start = std::chrono::high_resolution_clock::now();
        kernel_wrapper(d_matrix_a, d_matrix_b, d_matrix_c);
        cudaDeviceSynchronize();
        auto const end = std::chrono::high_resolution_clock::now();
        get_product_from_device();
        get_results(title, start, end);
    }

    void get_results( std::string title, std::chrono::time_point<std::chrono::high_resolution_clock> start, 
                      std::chrono::time_point<std::chrono::high_resolution_clock> end ) {
        std::cout << std::format("--------{}--------", title) << std::endl;
        #ifdef NO_OPENBLAS
            std::vector<R> matrix_c_expected = naive_cpu_matmul();
            if (print_result) std::cout << "Expected Result:" << std::endl;
            print_matrix<R>( matrix_c_expected, n, print_result );
            if (print_result) std::cout << "Actual Result:" << std::endl;
            print_matrix<R>( matrix_c, n, print_result );
            std::cout << "Correct Output:" << matrix_c == matrix_c_expected << std::endl;
        #else
            const std::vector<R> matrix_c_expected = cblas_cpu_matmul();
            if (print_result) std::cout << "Expected Result:" << std::endl;
            print_matrix<R>( matrix_c_expected, n, print_result );
            if (print_result) std::cout << "Actual Result:" << std::endl;
            print_matrix<R>( matrix_c, n, print_result );
            bool equal = matrix_c == matrix_c_expected;
            std::cout << "Correct Output:" << equal << std::endl;
        #endif

        std::cout << std::format("Time: {} us", 
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) 
                  << std::endl;
    }

    // Static Functions
    template< typename T >
    void print_matrix( const std::vector<T>& matrix, std::size_t side_length, bool enabled = false )
    {
        assert( side_length*side_length == matrix.size() && "matrix must be of length n*n");

        if (enabled)
        {
            for (std::size_t idx = 0; idx < n; ++idx)
            {
                std::cout << idx << ": ";
                for (std::size_t jdx = 0; jdx < n; ++jdx)
                {   
                    /* Note: Certain types like half do not have an << overload defined, so 
                       we cast to a float to ensure we can print.
                       May need to come up with a better/safer option later.
                    */
                    std::cout << static_cast<float> (matrix[idx*n + jdx]) << " ";
                }
                std::cout << std::endl;
            }
        }
    }

private:

    // Static Functions
    template< typename T >
    static std::vector<T> generate_matrix( T upper_bound, std::size_t n, std::size_t padded_n, uint32_t seed )
    {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<std::mt19937::result_type> distribution(0, upper_bound);

        std::vector<T> matrix;
        for ( std::size_t idx = 0; idx < padded_n * padded_n; ++idx ){
            std::size_t row = idx / padded_n;
            std::size_t col = idx % padded_n;
            T val = 0;
            std::size_t count = 0;
            if ((count++) < n*n && row < n && col < n){
                val = (T)distribution(rng);
            }
            matrix.push_back( val );
        }

        return matrix;
    }

    // Static Member Variables
    const uint32_t fixed_seed;

    constexpr std::size_t get_padded_sz( std::size_t n, std::size_t multiple)
    {
        return n%multiple ? n + (multiple - n%multiple) : n;
    }

    // Member Functions
    #ifdef NO_OPENBLAS

    std::vector<R> naive_cpu_matmul()
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

    std::vector<R> cblas_cpu_matmul()
    {   
        // Generate vectors of floats to be compatible with cblas
        const std::vector<float> matrix_a_float( matrix_a.begin(), matrix_a.end() );
        const std::vector<float> matrix_b_float( matrix_b.begin(), matrix_b.end() );

        // Create a zero-initialized result matrix of size nx n.
        std::vector<float> result( n * n, 0 );
        cblas_sgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0,
                    matrix_a_float.data(), n, matrix_b_float.data(), n, 1.0, result.data(), n );
        
        std::vector<R> result_converted( result.begin(), result.end() ); 
        
        return result_converted;
    }

    #endif

}; // class GemmExperiment


} // namespace a4
} // namepace csc485b