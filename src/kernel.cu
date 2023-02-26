#include "kernel.h"
#include "utils.h"

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void saturate_component(unsigned char *d_img, const size_t size, const size_t index)
{
   size_t block_size = blockDim.x * blockDim.y;
   size_t block_id = gridDim.x * blockIdx.y + blockIdx.x;
   size_t thread_id = blockDim.x * threadIdx.y + threadIdx.x;
   size_t id = block_id * block_size + thread_id;

   if (id < size)
      d_img[id * N_COMPONENT + index] = FULL;
}
