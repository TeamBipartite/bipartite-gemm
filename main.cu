/**
 * Driver for the benchmark comparison. Generates random data,
 * runs the CPU baseline, and then runs your code.
 */

#include <chrono>   // for timing
#include <iostream> // std::cout, std::endl
#include <iterator> // std::ostream_iterator
#include <vector>

#ifndef NO_OPENBLAS
#include <cblas.h>
#endif

#include "dense_graph.h"
#include "sparse_graph.h"

#include "data_generator.h"
#include "data_types.h"

/** 
 * Pads the given n to the least multiple of 32 not less than n
 */
constexpr std::size_t get_padded_sz(std::size_t n)
{
    return n%32 ? n + (32 - n%32) : n;
}

/**
 * Runs timing tests on a CUDA graph implementation.
 * Consists of independently constructing the graph and then
 * modifying it to its two-hop neighbourhood.
 */
template < typename DeviceGraph >
void run( DeviceGraph *g, csc485b::a2::edge_t const * d_edges, std::size_t m, std::size_t n )
{
    cudaDeviceSynchronize();
    auto const build_start = std::chrono::high_resolution_clock::now();

    // this code doesn't work yet!
    //csc485b::a2::gpu::build_graph<<< 1, 1 >>>( g, d_edges, m );
    csc485b::a2::gpu::build_graph( g, d_edges, m, n );

    cudaDeviceSynchronize();
    auto const reachability_start = std::chrono::high_resolution_clock::now();

    // neither does this!
    csc485b::a2::gpu::two_hop_reachability<<< dim3{n/32, n/32, n/32}, dim3{32, 32, 1} >>>( g );

    cudaDeviceSynchronize();
    auto const end = std::chrono::high_resolution_clock::now();

    std::cout << "Build time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(reachability_start - build_start).count()
              << " us"
              << std::endl;


    std::cout << "Reachability time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - reachability_start).count()
              << " us"
              << std::endl;
}

/**
 * Allocates space for a dense graph and then runs the test code on it.
 * Note that res is a float* so that it can be used with BLAS libraries
 */
void run_dense( csc485b::a2::edge_t const * d_edges, std::size_t n, std::size_t m,  float* res)
{
    using namespace csc485b;

    // allocate device DenseGraph
    a2::node_t * d_matrix, *d_dest;
    cudaMalloc( (void**)&d_matrix, sizeof( a2::node_t ) * n * n );
    cudaMalloc( (void**)&d_dest, sizeof( a2::node_t ) * n * n );
    a2::DenseGraph dg{ n, d_matrix, d_dest };
    a2::DenseGraph *d_dg;
    cudaMalloc( (void**)&d_dg, sizeof( a2::DenseGraph ) );
    cudaMemcpy( d_dg, &dg, sizeof( a2::DenseGraph ), cudaMemcpyHostToDevice );

    run( d_dg, d_edges, m, n );

    // check output?
    std::vector< a2::node_t > host_matrix( dg.matrix_size() );
    std::vector< a2::node_t > host_dest( dg.matrix_size() );
    a2::DenseGraph dg_res{ n, host_matrix.data(), host_dest.data() };
    cudaMemcpy( dg_res.adjacencyMatrix, dg.adjacencyMatrix, sizeof( a2::node_t ) * dg.matrix_size(), cudaMemcpyDeviceToHost );
    cudaMemcpy( dg_res.dest, dg.dest, sizeof( a2::node_t ) * dg.matrix_size(), cudaMemcpyDeviceToHost );
    for (int idx = 0; idx < n; idx++)
    {
        //std::cout << idx << ": ";
        for (int jdx = 0; jdx < n; jdx++)
        {
            //std::cout << dg_res.dest[idx*n + jdx] << " ";
        }
        //std::cout << "\n";
    }

    bool check = true;
    for (int idx = 0; idx < n; idx++)
    {
        for (int jdx = 0; jdx < n; jdx++)
        {
            if (dg_res.dest[idx*n + jdx]*1.0 != res[idx*n + jdx]){
                check = false;
                break;
            }
        }
    }

    std::cout << "Correct output: " << check << "\n";

    // clean up
    cudaFree( d_matrix );
}

/**
 * Allocates space for a sparse graph and then runs the test code on it.
 */
void run_sparse( csc485b::a2::edge_t const * d_edges, std::size_t n, std::size_t m )
{
    using namespace csc485b;

    // allocate device SparseGraph
    a2::node_t * d_offsets, * d_neighbours;
    cudaMalloc( (void**)&d_offsets,    sizeof( a2::node_t ) * (n+1) );
    cudaMalloc( (void**)&d_neighbours, sizeof( a2::node_t ) * m );
    a2::SparseGraph sg{n, m, d_offsets, d_neighbours };
    a2::SparseGraph *d_sg;
    cudaMalloc( (void**)&d_sg, sizeof( a2::SparseGraph ) );
    cudaMemcpy( d_sg, &sg, sizeof( a2::SparseGraph ), cudaMemcpyHostToDevice );
    //a2::SparseGraph d_sg{ n, m, d_offsets, d_neighbours };

    run( d_sg, d_edges, m, n );

    // check output
    a2::SparseGraph *sg_res;
    a2::node_t *offsets, *neighbours;
    sg_res = (a2::SparseGraph*)malloc(sizeof( a2::SparseGraph));
    offsets = (a2::node_t*)malloc(sizeof( a2::node_t) * n);
    neighbours = (a2::node_t*)malloc(sizeof( a2::node_t) * m);
    cudaMemcpy( sg_res, d_sg, sizeof( a2::SparseGraph ), cudaMemcpyDeviceToHost );
    cudaMemcpy( offsets, sg_res->neighbours_start_at, sizeof( a2::node_t ) * (n+1), cudaMemcpyDeviceToHost );
    cudaMemcpy( neighbours, d_neighbours, sizeof( a2::node_t ) * m, cudaMemcpyDeviceToHost );

    std::cout << "m: " << sg_res->m << " n: " << sg_res->n << "\n";

    for (int idx = 0; idx < n+1; idx++)
    {
        std::cout << offsets[idx] << " ";
    }
    std::cout << "\n";

    for (int idx = 0; idx < m; idx++)
    {
        std::cout << neighbours[idx] << " ";
    }
    std::cout << "\n";

    // clean up
    cudaFree( d_neighbours );
    cudaFree( d_offsets );
    cudaFree( d_sg );
    free(offsets);
    free(neighbours);
}

void matmul(float *mat, float *res, std::size_t n)
{
    for (std::size_t idx = 0; idx < n; idx++)
        for (std::size_t jdx = 0; jdx < n; jdx++)
            for (std::size_t kdx = 0; kdx < n; kdx++)
               res[idx*n + jdx] += mat[idx*n + kdx] * mat[kdx*n + jdx];
}

void print_matrix(float *matrix, std::size_t n){
      for (int row = 0; row < n; ++row){
        for (int col = 0; col < n; ++col){
            std::cout << matrix[(row * n) + col] << " ";
        }
        std::cout << "\n";
    }
}

void clamp(float *mat, std::size_t n)
{
    for (std::size_t idx = 0; idx < n; idx++)
    {
        for (std::size_t jdx = 0; jdx < n; jdx++)
        {
            if (idx == jdx || !mat[idx*n + jdx])
                mat[idx*n + jdx] = 0.0;
            else
                mat[idx*n + jdx] = 1.0;
        }
    }
}

int main()
{
    using namespace csc485b;
    
    // Create input
    std::size_t constexpr n = 4096;
    std::size_t constexpr expected_degree = n >> 2;

    a2::edge_list_t const graph = a2::generate_graph( n, n * expected_degree );
    std::size_t const m = graph.size();

    std::size_t padded_n = get_padded_sz(n);

    // lazily echo out input graph
    /*
    for( auto const& e : graph )
    {
        std::cout << "(" << e.x << "," << e.y << ") ";
    }
    std::cout << "\n";
    */

    // need to malloc since the matrix will exceed default stack size when n >= 1024
    float *matrix, *res;
    matrix = (float*) malloc(sizeof(float) * padded_n * padded_n);
    res = (float*) malloc(sizeof(float) * padded_n * padded_n);

    for (std::size_t idx = 0; idx < n*n; idx++)
    {
        matrix[idx] = 0.0;
        res[idx] = 0.0;
    }

    for ( auto const& e : graph ) {
        matrix[(e.x*padded_n) + e.y] = 1.0;
    }

    //print_matrix(matrix, padded_n);

    // OpenBLAS implementation
    auto const reachability_start = std::chrono::high_resolution_clock::now();

#ifdef NO_OPENBLAS
    // naive n^3 implementation
    matmul(matrix, res, padded_n);
#else
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, padded_n, padded_n, padded_n, 1.0,
                matrix, padded_n, matrix, padded_n, 1.0, res, padded_n);
#endif

    clamp(res, padded_n);
    auto const end = std::chrono::high_resolution_clock::now();

    std::cout << "Reachability time (CPU): "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - reachability_start).count()
              << " us"
              << std::endl;

    //std::cout << "Correct output\n";
    //print_matrix(res, padded_n);

    // allocate and memcpy input to device
    a2::edge_t * d_edges;
    cudaMalloc( (void**)&d_edges, sizeof( a2::edge_t ) * m );
    cudaMemcpyAsync( d_edges, graph.data(), sizeof( a2::edge_t ) * m, cudaMemcpyHostToDevice );

    // run your code!
    //run_sparse( d_edges, padded_n, m );
    //for (int idx = 0; idx < 10; idx++ )
        run_dense ( d_edges, padded_n, m, res );

    free(res);
    free(matrix);
    return EXIT_SUCCESS;
}
