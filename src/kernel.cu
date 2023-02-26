#include "kernel.h"
#include "utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

__device__ size_t index()
{
   size_t block_size = blockDim.x * blockDim.y;
   size_t block_id = gridDim.x * blockIdx.y + blockIdx.x;
   size_t thread_id = blockDim.x * threadIdx.y + threadIdx.x;
   size_t id = block_id * block_size + thread_id;

   return id;
}

__global__ void saturate_component(unsigned char *d_img, const size_t size,
                                   const size_t component_index)
{
   size_t id = index();

   if (id < size)
      d_img[id * N_COMPONENT + component_index] = FULL;
}

__global__ void horizontal_flip_kernel(unsigned char *d_img, const unsigned char *d_tmp,
                                       const size_t width, const size_t size)
{
   size_t id = index();

   if (id < size) {
      size_t row = id / width;
      size_t col = id % width;
      for (size_t i = 0; i < N_COMPONENT; ++i)
         d_img[(row * width + (width - col - 1)) * N_COMPONENT + i] = d_tmp[id * N_COMPONENT + i];
   }
}

__global__ void vertical_flip_kernel(unsigned char *d_img, const unsigned char *d_tmp,
                                     const size_t size)
{
   size_t id = index();

   if (id < size) {
      for (size_t i = 0; i < N_COMPONENT; ++i)
         d_img[id * N_COMPONENT + i] = d_tmp[(size - id) * N_COMPONENT + i]; // Flip vertical
   }
}