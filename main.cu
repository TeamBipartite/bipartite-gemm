/**
 * Driver for the benchmark comparison. Generates random data,
 * runs the CPU baseline, and then runs your code.
 */

#include <chrono>   // for timing
#include <iostream> // std::cout, std::endl
#include <iterator> // std::ostream_iterator
#include <vector>

#include "dense_graph.h"
#include "sparse_graph.h"

#include "data_generator.h"
#include "data_types.h"

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
 */
void run_dense( csc485b::a2::edge_t const * d_edges, std::size_t n, std::size_t m,  csc485b::a2::node_t* res)
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
    cudaMemcpy( dg_res.adjacencyMatrix, dg.dest, sizeof( a2::node_t ) * dg.matrix_size(), cudaMemcpyDeviceToHost );
    //std::copy( host_matrix.cbegin(), host_matrix.cend(), std::ostream_iterator< a2::node_t >( std::cout, " " ) );
    for (int idx = 0; idx < n; idx++)
    {
        std::cout << idx << ": ";
        for (int jdx = 0; jdx < n; jdx++)
        {
            std::cout << dg_res.adjacencyMatrix[idx*n + jdx] << " ";
        }
        std::cout << "\n";
    }

    bool check = true;
    for (int idx = 0; idx < n; idx++)
    {
        for (int jdx = 0; jdx < n; jdx++)
        {
            if (dg_res.adjacencyMatrix[idx*n + jdx] != res[idx*n + jdx]){
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
        //std::cout << offsets[idx] << " ";
    }
    std::cout << "\n";

    for (int idx = 0; idx < m; idx++)
    {
        //std::cout << neighbours[idx] << " ";
    }
    std::cout << "\n";

    // clean up
    cudaFree( d_neighbours );
    cudaFree( d_offsets );
    cudaFree( d_sg );
    free(offsets);
    free(neighbours);
}

void matmul(csc485b::a2::node_t *mat, csc485b::a2::node_t *res, std::size_t n)
{
    for (std::size_t idx = 0; idx < n; idx++)
        for (std::size_t jdx = 0; jdx < n; jdx++)
            for (std::size_t kdx = 0; kdx < n; kdx++)
               res[idx*n + jdx] += mat[idx*n + kdx] * mat[kdx*n + jdx];
}

void print_matrix(csc485b::a2::node_t * matrix, std::size_t n){
      for (int row = 0; row < n; ++row){
        for (int col = 0; col < n; ++col){
            std::cout << matrix[(row * n) + col] << " ";
        }
        std::cout << "\n";
    }
}

int main()
{
    using namespace csc485b;
    
    // Create input
    std::size_t constexpr n = 4096;
    std::size_t constexpr expected_degree = n >> 1;

    a2::edge_list_t const graph = a2::generate_graph( n, n * expected_degree );
    std::size_t const m = graph.size();

    // lazily echo out input graph
    for( auto const& e : graph )
    {
        //std::cout << "(" << e.x << "," << e.y << ") ";
    }
    std::cout << "\n";

    csc485b::a2::node_t *matrix, *res;
    matrix = (csc485b::a2::node_t*) malloc(sizeof(csc485b::a2::node_t) * n * n);
    res = (csc485b::a2::node_t*) malloc(sizeof(csc485b::a2::node_t) * n * n);
    //[n*n];
    //csc485b::a2::node_t res[n*n];
    for (std::size_t idx = 0; idx < n*n; idx++)
    {
        matrix[idx] = 0;
        res[idx] = 0;
    }

    /*
    for (int i = 0; i < m; i+= 2){
        auto e = graph[i];
        matrix[(e.x*n) + e.y] = 1;
        matrix[(e.y*n) + e.x] = 2;
    }
    */



    for ( auto const& e : graph ) {
        matrix[(e.x*n) + e.y] = 1;
    }

    //print_matrix(matrix, n);

    matmul(matrix, res, n);

    std::cout << "Correct output\n";
    print_matrix(res, n);

    

    // allocate and memcpy input to device
    a2::edge_t * d_edges;
    cudaMalloc( (void**)&d_edges, sizeof( a2::edge_t ) * m );
    cudaMemcpyAsync( d_edges, graph.data(), sizeof( a2::edge_t ) * m, cudaMemcpyHostToDevice );

    // run your code!
     run_sparse( d_edges, n, m );
    run_dense ( d_edges, n, m, res );

    free(res);
    free(matrix);
    return EXIT_SUCCESS;
}
