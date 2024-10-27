/**
 * The file in which you will implement your SparseGraph GPU solutions!
 */

#include <cstddef>  // std::size_t type

#include "cuda_common.h"
#include "data_types.h"

namespace csc485b {
namespace a2      {

/**
 * A SparseGraph is optimised for a graph in which the number of edges
 * is close to cn, for a small constanct c. It is represented in CSR format.
 */
struct SparseGraph
{
  std::size_t n; /**< Number of nodes in the graph. */
  std::size_t m; /**< Number of edges in the graph. */
  node_t * neighbours_start_at; /** Pointer to an n=|V| offset array */
  node_t * neighbours; /** Pointer to an m=|E| array of edge destinations */
};


namespace gpu {


__global__
void histogram( edge_t const *arr, std::size_t m, SparseGraph *g)
{
    const std::size_t global_th_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_th_id < g->m)
    {
      const node_t incident_vertex = arr[global_th_id].x;
      if (incident_vertex < g->n)
      {
        //g->neighbours_start_at[incident_vertex + 1] = 1;
        atomicAdd(g->neighbours_start_at + incident_vertex + 1, 1);
      }
    }

    return;
}

/* IN-PLACE prefix sum
 * TODO: move to header file
 */

template<typename T>
__global__
void prefix_sum( DenseGraph *g, T *block_sums, std::size_t n )
{
    const std::size_t global_th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t th_idx = threadIdx.x;

    __shared__ T smem[1024];
    smem[th_idx] = g->adjacencyMatrix[global_th_idx % n];
    __syncthreads();

    for (std::size_t stride = 1; stride < 1024; stride <<= 1)
    {
        std::size_t val = 0;

        if ( th_idx >= stride ){
            val = smem[th_idx - stride];
        }

        /*
          Needs to be a sync here.
          A fench only guarantees a strong ordering and flushing. But it does not guarantee
          that all threads will see the flushed value before it is accessed.
        */
        __syncthreads(); // Maybe can use a fence ?? -- Confirm with Sean

        if ( th_idx >= stride ){

            smem[th_idx] +=  val;
        }
         __syncthreads();

    }

    if (global_th_idx < n ){
        g->adjacencyMatrix[global_th_idx] = smem[th_idx];
    }
    __syncthreads();

    // Since we only launch a 1D kernel, the blockIdx.x's are unique
    if ( global_th_idx < n && (global_th_idx % 1024 == 1023 || global_th_idx == n -1 )  ){
        block_sums[blockIdx.x] = smem[th_idx];
    }


    return;
}


template<typename T>
__global__
void prefix_sum( SparseGraph *g, T *block_sums, std::size_t n )
{
    const std::size_t global_th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t th_idx = threadIdx.x;

    __shared__ T smem[1024];
    smem[th_idx] = g->neighbours_start_at[global_th_idx % n];
    __syncthreads();

    for (std::size_t stride = 1; stride < 1024; stride <<= 1)
    {
        std::size_t val = 0;

        if ( th_idx >= stride ){
            val = smem[th_idx - stride];
        }

        /*
          Needs to be a sync here.
          A fench only guarantees a strong ordering and flushing. But it does not guarantee
          that all threads will see the flushed value before it is accessed.
        */
        __syncthreads(); // Maybe can use a fence ?? -- Confirm with Sean

        if ( th_idx >= stride ){

            smem[th_idx] +=  val;
        }
         __syncthreads();

    }

    if (global_th_idx < n ){
        g->neighbours_start_at[global_th_idx] = smem[th_idx];
    }
    __syncthreads();

    // Since we only launch a 1D kernel, the blockIdx.x's are unique
    if ( global_th_idx < n && (!((global_th_idx+1) % 1024) || global_th_idx == n -1 )  ){
        block_sums[blockIdx.x] = smem[th_idx];
    }


    return;
}


__global__
void single_block_prefix_sum( node_t *arr, std::size_t n  ){
    const std::size_t th_id = threadIdx.x;

    /*
    Since we only launch 1 block, we can use all of smem.
    Tesla T4's have 64KB of smem per SM which gives 64KB/4B = 16384 uint32's
    But, using this much results in the following error: 
      function '_Z23single_block_prefix_sumIiEvPT_m' uses too much shared data (0x10000 bytes, 0xc000 max)
    So, only use half of 0xc000 bytes (i.e. 49152 bytes) per SM.
    49152B/4B = 12,288 unint32's 
    Use 12,288B/2 = 6144B for smem and other 6144B for scratch.
    */
    __shared__ node_t smem[6144];
    __shared__ node_t scratch[6144];

    // Put all values this thread is responsible for in smem and scratch
    for (std::size_t data_idx = th_id; data_idx < n; data_idx += 1024){
        const node_t val = arr[data_idx];
        smem[data_idx] = val;
        scratch[data_idx] = val;
    }

    __syncthreads();

    for (std::size_t stride = 1; stride < n; stride <<= 1){

        // copy into scratch
        for (std::size_t data_idx = th_id; data_idx < n; data_idx += 1024 ){
            scratch[data_idx] = smem[data_idx];
        }

        __syncthreads();

        for (std::size_t my_lane = th_id; my_lane < n; my_lane += 1024){

          std::size_t val = 0;

          if ( my_lane >= stride ){
              val = scratch[my_lane - stride];
          }

          __syncthreads();

          if ( my_lane >= stride ){
              smem[my_lane] +=  val;
          }
          __syncthreads();
        }
    }

    // Put all values this thread is responsible for back into the array.
    for (std::size_t data_idx = th_id; data_idx < n; data_idx += 1024){
        arr[data_idx] = smem[data_idx];
    }
}

template<typename T>
__global__
void prefix_sum_naive( T *arr, std::size_t n, std::size_t stride)
{
    const std::size_t th_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (th_idx < n && th_idx >= stride){
        arr[th_idx] +=  arr[th_idx - stride];
    }

    return;
}

template<typename T>
 __global__
void finish_prefix_sum(DenseGraph *g, T* block_sums, std::size_t n)
{
    const std::size_t th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx < n && blockIdx.x > 0){
        g->adjacencyMatrix[th_idx] += block_sums[blockIdx.x -1 ];
    }

    return;
}


template<typename T>
 __global__
void finish_prefix_sum(T *arr, T* block_sums, std::size_t n)
{
    const std::size_t th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx < n && blockIdx.x > 0){
        arr[th_idx] += block_sums[blockIdx.x -1 ];
    }

    return;
}


__global__
void store( SparseGraph *g, const edge_t *edges, node_t *scratch)
{
    const std::size_t global_th_id = blockIdx.x * blockDim.x + threadIdx.x;
    scratch[global_th_id] = g->neighbours_start_at[global_th_id];

    if (global_th_id < g->m)
    {
        edge_t edge = edges[global_th_id];
        node_t this_vertex  = edge.x;
        node_t other_vertex = edge.y;

        int pos = atomicAdd(scratch + this_vertex, 1);
        g->neighbours[pos] = other_vertex;
    }

    return;
}

// Emily
__global__
void invert_neighbours( SparseGraph *g, node_t *alpha);

// John
__global__
void populate_bitstring( SparseGraph *g, node_t *alpha);

// John
__global__
// Need to ask about whether new array of counts <= m ?
void get_global_counts( SparseGraph *g, uint8_t* bitstrings );

// Emily
__global__
void bitstring_store( SparseGraph *g, uint8_t* bitstrings );


/**
 * Constructs a SparseGraph from an input edge list of m edges.
 *
 * @pre The pointers in SparseGraph g have already been allocated.
 */
//__global__
void build_graph( SparseGraph *g, edge_t const * edge_list, std::size_t m, std::size_t n )
{
    std::size_t const threads_per_block = 1024;
    std::size_t const num_blocks =  ( n + threads_per_block - 1 ) / threads_per_block;

    node_t *tmp_blk_sums, *tmp_prefix_sums;
    cudaMalloc( (void**) &tmp_blk_sums, sizeof(node_t) * num_blocks );

    histogram<<< num_blocks, threads_per_block >>>( edge_list, m, g );


    // prefix_sum
    prefix_sum<<< num_blocks, threads_per_block >>>( g, tmp_blk_sums, n+1 );
    //single_block_prefix_sum<<< 1 , threads_per_block >>>( tmp_blk_sums, num_blocks );
    //finish_prefix_sum<<< num_blocks, threads_per_block >>>( tmp_blk_sums, tmp_blk_sums, input.size() );

    cudaFree(tmp_blk_sums);


    // store
    cudaMalloc( (void**) &tmp_prefix_sums, sizeof(node_t) * n );
    store<<< num_blocks, threads_per_block >>>(  g, edge_list,
                                                       tmp_prefix_sums );

    cudaFree( (void**) tmp_prefix_sums);
    return;
}

/**
  * Repopulates the adjacency lists as a new graph that represents
  * the two-hop neighbourhood of input graph g
  */
__global__
void two_hop_reachability( SparseGraph *g )
{
    // TODO: Cannot be a device function
    // TODO: Update dense so functions match
    return;
}

} // namespace gpu
} // namespace a2
} // namespace csc485b
