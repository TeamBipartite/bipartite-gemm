#include <cstddef>
#include <mma.h>

using namespace nvcuda;

namespace csc485b {
namespace a4 {

namespace tensorcores{

/** half_gemm
  * @brief perform a gemm on two fp16 matricies using tensor wmma instructions
  * @pre maxtrix_a, matrix_b, and result are n x n matricies
  */
__global__
void half_gemm(half *matrix_a, half *matrix_b, half *res, std::size_t n)
{
    // TODO: parameterize or templetize this
    const int WMMA_M = 16;
    const int WMMA_K = 16;
    const int WMMA_N = 16;

    // Note that threadblocks are a 4x4 2D grid of warps
    // Z-dimension only at block-level
    std::size_t a_col = 0; //((blockIdx.z * blockDim.x + threadIdx.x) / 32) * WMMA_M;
    const std::size_t a_row = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_K;

    const std::size_t b_col = ((blockIdx.x * blockDim.x + threadIdx.x) / 32) * WMMA_K;
    std::size_t b_row = 0; //(blockIdx.z * blockDim.y + threadIdx.y) * WMMA_N;

    const std::size_t c_col = ((blockIdx.z * blockDim.x + threadIdx.x) / 32) * WMMA_M;
    const std::size_t c_row = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_N;
    
    if (a_row >= n || b_col >= n) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> afrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_K, WMMA_N, half, wmma::row_major> bfrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_K, WMMA_N, half>  acc;

    wmma::load_matrix_sync(acc, res + c_row * n + c_col, n, wmma::mem_row_major);

    for (std::size_t z = 0; z < n/WMMA_N; z++)
    {
        wmma::load_matrix_sync(afrag, matrix_a + a_row * n + a_col, n);
        wmma::load_matrix_sync(bfrag, matrix_b + b_row * n + b_col, n);
        wmma::mma_sync(acc, afrag, bfrag, acc);
        a_col += WMMA_M;
        b_row += WMMA_N;
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

namespace tensorcores{

using namespace nvcuda;

__global__
void transpose_matrix(half* matrix, half* matrix_transpose, std::size_t n) {
  const std::size_t th_idx = blockDim.x * blockIdx.x + threadIdx.x;
  const std::size_t row = th_idx / n;
  const std::size_t col = th_idx % n;

  if (th_idx < n){
    matrix_transpose[(col * n) + row] = matrix[(row * n) + col];
  }

  return;

}

__global__
void fp32_wmma_gemm( half* matrix_a, half* matrix_b, float* result, std::size_t n ){
  const std::size_t tile_side_len = 16;
  const std::size_t lda = n;
  const std::size_t ldb = n;
  const std::size_t ldc = n;

  /*
   This block is a grid of 4x4 warps.
   Assign this warp to a unique 16x16 tile in matrix_a and matrix_b
  */
  // Compute warp row and column within the grid
  const std::size_t warp_col = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const std::size_t warp_row = (blockIdx.y * blockDim.y) + threadIdx.y;

  // Declare fragments
  // Fragments for matrix a and matrix b are FP16
  wmma::fragment<wmma::matrix_a, tile_side_len, tile_side_len, tile_side_len, half, wmma::row_major> a_frag; 
  wmma::fragment<wmma::matrix_b, tile_side_len, tile_side_len, tile_side_len, half, wmma::row_major> b_frag; // TODO: transpose B
  // Fragments for accumulator are FP16
  wmma::fragment<wmma::accumulator, tile_side_len, tile_side_len, tile_side_len, float> accum_frag;
  wmma::fill_fragment(accum_frag, 0.0f);

  // Compute the matrix multiplication of a 64 x 64 tle of the result matrix
  for (std::size_t k = 0; k < n; k+= tile_side_len){

    const std::size_t a_row = warp_row * tile_side_len; 
    const std::size_t a_col = k;

    const std::size_t b_row = k;
    const std::size_t b_col = warp_col * tile_side_len;

    if (a_row < n && a_col < n && b_row < n && b_col < n) {
      wmma::load_matrix_sync(a_frag, matrix_a + (a_row * n) + a_col, lda);
      wmma::load_matrix_sync(b_frag, matrix_b + (b_row * n) + b_col, ldb);
    }

    // Compute matrix multiplication
    wmma::mma_sync(accum_frag, a_frag, b_frag, accum_frag);
  }

  // Store results back in matrix C
  const std::size_t c_row = warp_row * tile_side_len; 
  const std::size_t c_col = warp_col * tile_side_len;

  if (c_row < n && c_col < n){
    wmma::store_matrix_sync(result + (c_row * n) + c_col, accum_frag, ldc, wmma::mem_row_major);
  }


}

} // namespace tensorcores

} // namespace a4
} // namespace csc485b
