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

#define US_PER_S 1000000
#define GIGA     1000000000

namespace csc485b {
namespace a4 {

template < typename I, typename R >
class GemmExperiment {

public:

    // Member Variables
    const std::size_t n;

    const std::vector<I> matrix_a;
    I* d_matrix_a;
    I* h_matrix_a;

    const std::vector<I> matrix_b;
    I* d_matrix_b;
    I* h_matrix_b;

    std::vector<R> matrix_c;
    R* d_matrix_c;
    R* h_matrix_c;

    bool print_result;

    const std::size_t superblock_sz;
    std::vector<cudaStream_t> streams;


    // Member Functions
    GemmExperiment( std::size_t input_n, std::size_t upper_bound, std::size_t multiple, uint32_t seed, bool print_result ):
                                                        n{get_padded_sz(input_n, multiple)},
                                                        matrix_a{generate_matrix<I>(I(upper_bound), input_n, n, seed)}, 
                                                        matrix_b{generate_matrix<I>(I(upper_bound), input_n, n, seed)},
                                                        matrix_c{std::vector<R>(n*n, R(0))},
                                                        d_matrix_a{nullptr},
                                                        d_matrix_b{nullptr},
                                                        d_matrix_c{nullptr},
                                                        h_matrix_a{nullptr},
                                                        h_matrix_b{nullptr},
                                                        h_matrix_c{nullptr},
                                                        superblock_sz{n},
                                                        streams{nullptr},
                                                        fixed_seed{seed},
                                                        print_result{print_result} {}

    GemmExperiment( std::size_t input_n, std::size_t upper_bound, std::size_t multiple, uint32_t seed, bool print_result, std::size_t superblock_sz ):
                                                        n{get_padded_sz(input_n, multiple)},
                                                        matrix_a{generate_matrix<I>(I(upper_bound), input_n, n, seed)}, 
                                                        matrix_b{generate_matrix<I>(I(upper_bound), input_n, n, seed)},
                                                        matrix_c{std::vector<R>(n*n, R(0))},
                                                        d_matrix_a{nullptr},
                                                        d_matrix_b{nullptr},
                                                        d_matrix_c{nullptr},
                                                        h_matrix_a{nullptr},
                                                        h_matrix_b{nullptr},
                                                        h_matrix_c{nullptr},
                                                        superblock_sz{superblock_sz},
                                                        streams{std::vector<cudaStream_t>(2)},
                                                        fixed_seed{seed},
                                                        print_result{print_result} 
     {

         cudaStreamCreate(&streams[0]);
         cudaStreamCreate(&streams[1]);
     }
    
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
 
        // Copy contents of matrix_a and matrix_b to pinned memory
        cudaMallocHost((void**) &h_matrix_a, sizeof( I ) * matrix_a.size());
        cudaMallocHost((void**) &h_matrix_b, sizeof( I ) * matrix_b.size());
        cudaMallocHost((void**) &h_matrix_b, sizeof( R ) * matrix_c.size());
        memcpy(h_matrix_a, matrix_a.data(),  sizeof( I ) * matrix_a.size());
        memcpy(h_matrix_b, matrix_b.data(),  sizeof( I ) * matrix_b.size());
          // events for timing
          cudaEvent_t startEvent, stopEvent;

           cudaEventCreate(&startEvent) ;
           cudaEventCreate(&stopEvent) ;

          cudaEventRecord(startEvent, 0) ;
        cudaMemcpy( d_matrix_a, h_matrix_a, sizeof( I ) * matrix_a.size(), cudaMemcpyHostToDevice );
        cudaMemcpy( d_matrix_b, h_matrix_b, sizeof( I ) * matrix_b.size(), cudaMemcpyHostToDevice );
           cudaEventRecord(stopEvent, 0) ;
           cudaEventSynchronize(stopEvent);

          float time;
           cudaEventElapsedTime(&time, startEvent, stopEvent);
//           std::cout << "TRANSFER TIME: " << time << "\n";

        // Set contents of matrix_c to zero on device
        cudaMemset(d_matrix_c, 0x0, sizeof(R) * matrix_c.size() );
    }

    void get_product_from_device(){
        cudaMemcpy( matrix_c.data(), d_matrix_c, sizeof(R) * matrix_c.size(), cudaMemcpyDeviceToHost );
    }

    std::size_t get_n(){
        return n;
    }

    std::size_t get_superblk_sz(){
        return superblock_sz;
    }

    void run_experiment( std::function<void(I*, I*, R*)> kernel_wrapper, std::string title){
        auto const start = std::chrono::high_resolution_clock::now();
        prepare_device();
        kernel_wrapper(d_matrix_a, d_matrix_b, d_matrix_c);
        cudaDeviceSynchronize();
        auto const end = std::chrono::high_resolution_clock::now();
        get_product_from_device();
        get_results(title, start, end);
    }

    void run_experiment_streams( std::function<void(I*, I*, R*, std::size_t, cudaStream_t)> kernel_wrapper, std::string title)
    {
        //prepare_device();
        auto const start = std::chrono::high_resolution_clock::now();
        cudaMalloc( &d_matrix_a, sizeof( I ) * 2 * superblock_sz * n );
        cudaMalloc( &d_matrix_b, sizeof( I ) * n * n );
        cudaMalloc( &d_matrix_c, sizeof( R ) * 2 * superblock_sz * superblock_sz );
 
        // Copy contents of matrix_a and matrix_b to pinned memory
        cudaMallocHost((void**) &h_matrix_a, sizeof( I ) * matrix_a.size());
        cudaMallocHost((void**) &h_matrix_b, sizeof( I ) * matrix_b.size());
        cudaMallocHost((void**) &h_matrix_c, sizeof( R ) * matrix_b.size());
        memcpy(h_matrix_a, matrix_a.data(),  sizeof( I ) * matrix_a.size());
        memcpy(h_matrix_b, matrix_b.data(),  sizeof( I ) * matrix_b.size());
        cudaMemcpy( d_matrix_b, h_matrix_b,  sizeof( I ) * matrix_b.size(), cudaMemcpyHostToDevice );

        assert( n * n == matrix_a.size() && "GemmExperiment need to be of size n x n" );
        // PRECONDITION: superblock_sz is a factor of n
        assert(n % superblock_sz ==0 && "superblock_sz must be a factor of n");
        for (std::size_t i = 0; i < n/superblock_sz; i+=2)
        {
          cudaMemcpyAsync( d_matrix_a, h_matrix_a+superblock_sz*i*n, sizeof( I ) * superblock_sz*n, cudaMemcpyHostToDevice, streams[0] );
          cudaMemcpyAsync( d_matrix_a + superblock_sz*n, h_matrix_a+superblock_sz*(i+1)*n, sizeof( I ) * superblock_sz*n, cudaMemcpyHostToDevice, streams[1] );
          for (std::size_t j = 0; j < n/superblock_sz; j++)
          {
            kernel_wrapper(d_matrix_a, d_matrix_b, d_matrix_c, j, streams[0]);
            kernel_wrapper(d_matrix_a+superblock_sz*n, d_matrix_b, d_matrix_c+superblock_sz*superblock_sz, j, streams[1]);
            cudaMemcpyAsync( h_matrix_c + superblock_sz*(i)*n + superblock_sz*j, d_matrix_c+(superblock_sz)*superblock_sz, sizeof( R ) * superblock_sz * superblock_sz, cudaMemcpyDeviceToHost, streams[0] );
            cudaMemcpyAsync( h_matrix_c + superblock_sz*(i+1)*n + superblock_sz*j, d_matrix_c+(superblock_sz)*superblock_sz, sizeof( R ) * superblock_sz * superblock_sz, cudaMemcpyDeviceToHost, streams[1] );
            //for (std::size_t k = 0; k < superblock_sz; k++)
            //{
            //    cudaMemcpyAsync(  h_matrix_c + superblock_sz*i*n + k*n + superblock_sz*j, d_matrix_c + k*superblock_sz, sizeof( R ) * superblock_sz, cudaMemcpyDeviceToHost, streams[0] );
            //    cudaMemcpyAsync( h_matrix_c + superblock_sz*(i+1)*n + k*n + superblock_sz*j, d_matrix_c+(superblock_sz+k)*superblock_sz, sizeof( R ) * superblock_sz, cudaMemcpyDeviceToHost, streams[1] );
            //}
            //cudaMemset(d_matrix_c, 0x0, sizeof(R) * superblock_sz * superblock_sz * 2 );
            //cudaStreamSynchronize(streams[0]);
            //cudaStreamSynchronize(streams[1]);
//          cudaEvent_t startEvent, stopEvent;

//           cudaEventCreate(&startEvent) ;
//           cudaEventCreate(&stopEvent) ;

//          cudaEventRecord(startEvent, 0) ;

            //if (!i)
            //{

                //cudaMemcpyAsync( d_matrix_b, h_matrix_b, sizeof( I ) * n, cudaMemcpyHostToDevice, streams[0] );
            //}
//           cudaEventRecord(stopEvent, 0) ;
//           cudaEventSynchronize(stopEvent);

//          float time;
//           cudaEventElapsedTime(&time, startEvent, stopEvent);
//           std::cout << "TRANSFER TIME: " << time << "\n";
    
        // Set contents of matrix_c to zero on device
        //cudaMemset(d_matrix_c, 0x0, sizeof(R) * matrix_c.size() );
        //kernel_wrapper(d_matrix_a, d_matrix_b, d_matrix_c);

          }
        }
        cudaDeviceSynchronize();
        auto const end = std::chrono::high_resolution_clock::now();
        //get_product_from_device();
        memcpy(matrix_c.data(), h_matrix_c,  sizeof( I ) * matrix_c.size());
        get_results(title, start, end);
    }

    void get_results( std::string title, std::chrono::time_point<std::chrono::high_resolution_clock> start, 
                      std::chrono::time_point<std::chrono::high_resolution_clock> end ) {
        std::cout << std::format("--------{}--------", title) << std::endl;

#ifdef NO_OPENBLAS
        std::vector<R> matrix_c_expected = naive_cpu_matmul();
#else
        const std::vector<R> matrix_c_expected = cblas_cpu_matmul();
#endif

        if (print_result) std::cout << "Expected Result:" << std::endl;
        print_matrix<R>( matrix_c_expected, n, print_result );
        if (print_result) std::cout << "Actual Result:" << std::endl;
        print_matrix<R>( matrix_c, n, print_result );

        bool equal = matrix_c == matrix_c_expected;
        std::cout << "Correct Output:" << equal << std::endl;


        std::size_t time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        double gflops = get_gflops(time, 28);


        std::cout << std::format("Time: {} us", time) << std::endl
                  << std::format("Estimated GFLOPs: {}", gflops) << std::endl;
    }

    double get_gflops(std::size_t us, std::size_t num_sms)
    {
        double s = us*1.0 / US_PER_S;
        return 2 * (n*n*n) / s / GIGA / num_sms;
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
