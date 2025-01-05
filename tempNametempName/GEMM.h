#include <cstddef>
#include <mma.h>

#define WARP_SZ 32
#define WMMA_M  16
#define WMMA_K  16
#define WMMA_N  16

using namespace nvcuda;

namespace tempNametempName{
namespace tensorcores{

/** gemm
  * @brief perform a gemm on two matricies of type I using tensor wmma
  *        instructions, saving the results in the type R matrix
  * @pre matrix_a, matrix_b, and res are n x n matricies
  */
template<typename I, typename R>
__global__
void gemm(I *matrix_a, I *matrix_b, R *res, std::size_t n, std::size_t superblock_sz=0)
{

    // Note that threadblocks are a 4x4 2D grid of warps
    std::size_t a_col = 0; 
    const std::size_t a_row = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_K;

    const std::size_t b_col = ((blockIdx.x * blockDim.x + threadIdx.x) / WARP_SZ) * WMMA_K;
    std::size_t b_row = 0;

    const std::size_t c_col = ((blockIdx.x * blockDim.x + threadIdx.x) / WARP_SZ) * WMMA_M;
    const std::size_t c_row = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_N;

    // Safe as this will be consistent for an entire kernel launch
    const std::size_t num_rows = (superblock_sz) ? superblock_sz : n; 

    if (a_row >= num_rows || b_col >= n) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, I, wmma::row_major> afrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, I, wmma::row_major> bfrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, R> acc;
    wmma::fill_fragment(acc, R(0));

    for (std::size_t k = 0; k < n; k += WMMA_K)
    {
        a_col = k;
        b_row = k;
        wmma::load_matrix_sync(afrag, matrix_a + a_row * n + a_col, n);
        wmma::load_matrix_sync(bfrag, matrix_b + b_row * n + b_col, n);
        wmma::mma_sync(acc, afrag, bfrag, acc);
    }

    wmma::store_matrix_sync(res + c_row * n + c_col, acc, n, wmma::mem_row_major);
}

} // namespace tensorcores


namespace cudacores{

/**
 * warp_sum
 * @brief Perform a warp sum reduction using given th_val
 */
__device__
std::size_t warp_sum(std::size_t th_val)
{
  std::size_t th_id = threadIdx.x;
  std::size_t new_val = 0;
  uint32_t shuffle_mask = 0xFFFFFFFF;

  for (std::size_t stride = 1; stride < WARP_SZ; stride <<= 1)
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
  * matrix_mult
  * @brief Compute the partial product of a 32x32 tile of matrix_a and matrix_b, storing results in result matrix.
  * @pre matrix_a, matrix_b, and result have dimensions of n x n
*/
__global__
void matrix_mult( uint32_t* matrix_a, uint32_t* matrix_b, uint32_t* result, std::size_t n)
{
    // Remember: Multiple z dimensions at block-level ONLY

    // A
    std::size_t a_col = blockIdx.z * blockDim.x + threadIdx.x;
    std::size_t a_row = blockIdx.y * blockDim.y + threadIdx.y;

    // B
    std::size_t b_col = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t b_row = blockIdx.z * blockDim.y + threadIdx.y;

    // C
    //std::size_t c_col = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t c_row = blockIdx.y * blockDim.y + threadIdx.y;

    // Copy tile of B (transposed) into smem
    __shared__ uint32_t smem[1024];
    smem[(threadIdx.x * blockDim.x) + threadIdx.y ] = matrix_b[(b_row * n) + b_col];
    __syncthreads();

    // Each thread performs calculations for a fixed a value, retrieve it here
    std::size_t a_val = matrix_a[(a_row * n) + a_col];

    for (std::size_t b_tile_col = 0; b_tile_col < blockDim.x; b_tile_col++)
    {
      // Perform single cell product of a and b for thread
      std::size_t product =  a_val * smem[(b_tile_col * blockDim.x) + threadIdx.x];

      // Make sure that all accesses to smem are complete before we perform warp_sum
      __syncwarp();

      // Use warp primitives to add
      std::size_t dot_product = warp_sum(product);
      if (!threadIdx.x)
        atomicAdd(result + (c_row * n) + (blockIdx.x * blockDim.x) + b_tile_col, dot_product);
    }

    return;
}

template< typename T >
__global__
void mark_matrix_element( T* matrix, uint32_t* matrix_marks, std::size_t n ) // where n is the length of matrix and matrix_marks
{
    const std::size_t th_id = ( blockDim.x * blockIdx.x ) + threadIdx.x;

    if ( th_id < n && matrix[th_id] ){
        matrix_marks[th_id] = 1;
    }

}

/**
 * prefix_sum
 * @brief Performs a block-local prefix sums and stores final result to an array of size = # of blocks
 */
template<typename T>
__global__
void prefix_sum( T* arr, T* block_sums, std::size_t n )
{
    const std::size_t global_th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t th_idx = threadIdx.x;

    __shared__ T smem[1024];
    smem[th_idx] = arr[global_th_idx % n];
    __syncthreads();

    for (std::size_t stride = 1; stride < 1024; stride <<= 1)
    {
        std::size_t val = 0;

        if ( th_idx >= stride ){
            val = smem[th_idx - stride];
        }

        __syncthreads();

        if ( th_idx >= stride ){

            smem[th_idx] +=  val;
        }
         __syncthreads();

    }

    if (global_th_idx < n ){
        arr[global_th_idx] = smem[th_idx];
    }
    __syncthreads();

    // Since we only launch a 1D kernel, the blockIdx.x's are unique
    if ( global_th_idx < n && (!((global_th_idx+1) % 1024) || global_th_idx == n -1 )  ){
        block_sums[blockIdx.x] = smem[th_idx];
    }


    return;
}

/**
 * single_block_prefix_sum
 * @brief Performs a prefix sum in a single block
 */
template < typename T >
__global__
void single_block_prefix_sum( T* arr, std::size_t n, uint32_t* num_non_zeros_ptr = nullptr  ){
    const std::size_t th_id = threadIdx.x;
    std::size_t max_number_items = n / 1024;
    if ( n % 1024 ){
        ++max_number_items;
    }


    /*
    Since we only launch 1 block, we can use all of smem.
    Tesla T4's have 64KB of smem per SM which gives 64KB/4B = 16384 uint32's
    But, using this much results in the following error: 
      function '_Z23single_block_prefix_sumIiEvPT_m' uses too much shared data (0x10000 bytes, 0xc000 max)
    So, only use half of 0xc000 bytes (i.e. 49152 bytes) per SM.
    49152B/4B = 12,288 unint32s 
    Use 12,288B/2 = 6144 uint32s for smem and other 6144 uint32s for scratch.
    */
    __shared__ T smem[6144];
    __shared__ T scratch[6144];

    // Put all values this thread is responsible for in smem and scratch
    for (std::size_t data_idx = th_id; data_idx < n; data_idx += 1024){
        const T val = arr[data_idx];
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

        // All threads in the block need to do the same number of iterations to ensure they all reach the sync!
        for (std::size_t my_lane = th_id, iteration = 0; iteration < max_number_items; my_lane += 1024, ++iteration){

          std::size_t val = 0;

          if ( my_lane < n && my_lane >= stride ){
              val = scratch[my_lane - stride];
          }

          __syncthreads();

          if ( my_lane < n && my_lane >= stride ){
              smem[my_lane] +=  val;
          }
          __syncthreads();
        }
    }

    // Put all values this thread is responsible for back into the array.
    for (std::size_t data_idx = th_id; data_idx < n; data_idx += 1024){
        arr[data_idx] = smem[data_idx];
    }

    if ( num_non_zeros_ptr ){
        // The value of num_non_zeros_ptr is the same for all threads, so the sync in the if block is SAFE!
        __syncthreads();
        if (!th_id){
            *num_non_zeros_ptr = smem[n-1];
        }

    }
}

template < typename T >
__global__
void insert_non_zero_elements( T* input, uint32_t* positions, T* output, std::size_t n ) // where n is the size of the input
{
    const std::size_t th_id = ( blockDim.x * blockIdx.x ) + threadIdx.x;
    const T input_element = input[th_id % n];
    
    if (th_id < n && input_element ){
        output[ positions[th_id] ] = input_element;
    }
}

template < typename T >
__global__
void histogram( T* hist, std::size_t n ) // where matrix has n rows and n columns
{
    const std::size_t th_id = blockIdx.x * blockDim.x + threadIdx.x;

    if ( th_id < ( n * n ) )
    {
        const T row = th_id / n;
        atomicAdd( hist + row, 1 ); //or maybe atomicAdd( hist + row + 1, 1 ); if passing in hist + 1 doesn't work
    }

    return;
}

template < typename I, typename T >
__global__
void store( I* input, T* scratch, T* cols, std::size_t n )
{
    const std::size_t th_id = blockIdx.x * blockDim.x + threadIdx.x;

    if ( th_id < ( n * n ) && input[th_id])
    {
        const T row = th_id / n;
        const T col = th_id % n;

        int pos = atomicAdd(scratch + row, 1);
        cols[pos] = col;
    }

    return;
}


template < typename T >
__global__
void copy( T* input, T* copy, std::size_t n )
{
    const std::size_t th_id = blockIdx.x * blockDim.x + threadIdx.x;

    if ( th_id < n )
    {
        copy[th_id] = input[th_id];
    }

    return;
}




} // namespace cudacores
} // namespace tempNametempName
