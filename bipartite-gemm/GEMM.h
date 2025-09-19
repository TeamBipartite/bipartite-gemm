#include <cstddef>
#include <mma.h>

#define WARP_SZ 32
#define WMMA_M  16
#define WMMA_K  16
#define WMMA_N  16

using namespace nvcuda;

namespace bipartite{
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

} // namespace cudacores
} // namespace bipartite
