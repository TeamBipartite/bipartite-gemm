/**
 * The file in which you will implement your DenseGraph GPU solutions!
 */

#include <cstddef>  // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

namespace csc485b {
namespace a2      {

/**
 * A DenseGraph is optimised for a graph in which the number of edges
 * is close to n(n-1). It is represented using an adjacency matrix.
 */
struct DenseGraph
{
  std::size_t n; /**< Number of nodes in the graph. */
  node_t * adjacencyMatrix; /** Pointer to an n x n adj. matrix */
  node_t * dest;

  /** Returns number of cells in the adjacency matrix. */
  __device__ __host__ __forceinline__
  std::size_t matrix_size() const { return n * n; }
};


namespace gpu {

/**
 * Constructs a DenseGraph from an input edge list of m edges.
 *
 * @pre The pointers in DenseGraph g have already been allocated.
 */
__global__
void add_neighbour_pair( DenseGraph *g, edge_t const * edge_list, std::size_t m)
{

    const std::size_t th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Results in index access pattern of 0, 3, 4, 7, 8, ... to facilitate coalescing
    const std::size_t edge_idx = ( th_idx * 2 ) + ( th_idx % 2 );

    if ( edge_idx < m ){

        const int2 edge = edge_list[edge_idx];
        g->adjacencyMatrix[(edge.x * g->n) + edge.y] = 1;
        g->adjacencyMatrix[(edge.y * g->n) + edge.x] = 1;

    }

    return;
}

void build_graph( DenseGraph *g, edge_t const * edge_list, std::size_t m, std::size_t n )
{

    std::size_t const threads_per_block = 1024;
    std::size_t const num_blocks =  ( m + threads_per_block - 1 ) / threads_per_block;

    add_neighbour_pair<<< num_blocks, threads_per_block>>>(g, edge_list, m);
    return;
}

__device__
std::size_t warp_sum(std::size_t th_val)
{
  std::size_t th_id = threadIdx.x;
  std::size_t new_val = 0;
  uint32_t shuffle_mask = 0xFFFFFFFF;

  for (std::size_t stride = 1; stride < 32; stride <<= 1)
  {
      new_val = __shfl_down_sync(0xFFFFFFFF, th_val, stride);
      // Only add the new value if this thread is in the mask!
      if ((0x1 << th_id) & shuffle_mask){
        th_val += new_val;
      }
      shuffle_mask >>= stride;
  }

  return th_val;

}

/**
  * Repopulates the adjacency matrix as a new graph that represents
  * the two-hop neighbourhood of input graph g
  */
__global__
void two_hop_reachability_kernel( DenseGraph *g )
{
    // Remember: Multiple z dimensions at block-level ONLY

    // A
    std::size_t a_col = blockIdx.z * blockDim.x + threadIdx.x;
    std::size_t a_row = blockIdx.y * blockDim.y + threadIdx.y;

    // B
    std::size_t b_col = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t b_row = blockIdx.z * blockDim.y + threadIdx.y;

    // C
#ifdef USE_FULL_MULTIPLY
    std::size_t c_col = blockIdx.x * blockDim.x + threadIdx.x;
#endif
    std::size_t c_row = blockIdx.y * blockDim.y + threadIdx.y;

    // Copy tile of B (transposed) into smem
    __shared__ a2::node_t smem[1024];
    smem[(threadIdx.x * blockDim.x) + threadIdx.y ] = g->adjacencyMatrix[(b_row * g->n) + b_col];
    __syncthreads();

    // Each thread performs calculations for a fixed a value, retrieve it here
    a2::node_t a_val = g->adjacencyMatrix[ (a_row * g->n) + a_col];

    for (std::size_t b_tile_col = 0; b_tile_col < blockDim.x; b_tile_col++)
    {
      // Perform single cell product of a and b for thread
      std::size_t product =  a_val * smem[(b_tile_col * blockDim.x) + threadIdx.x];

      // make sure that all accesses to smem are complete before we perform warp_sum
      __syncwarp();

#ifdef USE_FULL_MULTIPLY
#ifdef NO_WARP_PRIMITIVES
      // use Atomics to add instead of warp primitives
      atomicAdd(g->dest + (c_row * g->n) + (blockIdx.x * blockDim.x) + b_tile_col, product);
#else
      // use warp primitives to add
      std::size_t dot_product = warp_sum(product);
      if (!threadIdx.x)
        atomicAdd(g->dest + (c_row * g->n) + (blockIdx.x * blockDim.x) + b_tile_col, dot_product);
#endif // NO_WARP_PRIMITIVES
#else // USE_FULL_MULTIPLY
      if (c_row != (blockIdx.x * blockDim.x + b_tile_col) && product)
        g->dest[(c_row * g->n) + (blockIdx.x * blockDim.x) + b_tile_col] = 1;
#endif

    }

#ifdef USE_FULL_MULTIPLY
    g->dest[(c_row * g->n) + c_col] = g->dest[(c_row * g->n) + c_col] && !(c_row == c_col);
#endif
    return;
}

/**
 * @brief wrapper host function for two_hop_reachability (needed since this cannot
 *        be a kerner for sparse
 */
void two_hop_reachability( DenseGraph *g, std::size_t n, std::size_t m )
{
    uint32_t block_dim_sz = (uint32_t)(n / 32);
    two_hop_reachability_kernel<<< dim3{block_dim_sz, block_dim_sz, block_dim_sz}, 
                                   dim3{32, 32, 1} >>>(g);
}

} // namespace gpu
} // namespace a2
} // namespace csc485b
