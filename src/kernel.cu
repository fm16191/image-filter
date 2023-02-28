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

__global__ void extract_component(unsigned char *d_img, const size_t size,
                                  const size_t component_index)
{
   size_t id = index();

   if (id < size)
      for (size_t i = 0; i < N_COMPONENT; ++i) {
         if (i == component_index)
            continue;
         d_img[id * N_COMPONENT + i] = 0;
      }
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
         d_img[id * N_COMPONENT + i] = d_tmp[(size - id) * N_COMPONENT + i];
   }
}

__global__ void blur_kernel(unsigned char *d_img, const unsigned char *d_tmp, const size_t height,
                            const size_t width)
{
   size_t id = index();

   size_t size = height * width;

   if (id < size) {
      for (size_t i = 0; i < N_COMPONENT; ++i) {
         size_t mean = (size_t)d_tmp[id * N_COMPONENT + i];
         if ((id + 1) % width != 0)
            mean += (size_t)d_tmp[(id + 1) * N_COMPONENT + i];
         if (id % width != 0)
            mean += (size_t)d_tmp[(id - 1) * N_COMPONENT + i];
         if (id > width)
            mean += (size_t)d_tmp[(id - width) * N_COMPONENT + i];
         if (id < size - width)
            mean += (size_t)d_tmp[(id + width) * N_COMPONENT + i];
         d_img[id * N_COMPONENT + i] = (unsigned char)(mean / 5);
      }
   }
}

__global__ void grayscale_kernel(unsigned char *d_img, const size_t size)
{
   size_t id = index();

   if (id < size) {
      unsigned char rgb = (unsigned char)((float)d_img[id * N_COMPONENT + 0] * 0.299f +
                                          (float)d_img[id * N_COMPONENT + 1] * 0.587f +
                                          (float)d_img[id * N_COMPONENT + 2] * 0.114f);
      for (size_t i = 0; i < N_COMPONENT; ++i)
         d_img[id * N_COMPONENT + i] = rgb;
   }
}

__global__ void negative_kernel(unsigned char *d_img, const size_t size)
{
   size_t id = index();

   if (id < size) {
      for (size_t i = 0; i < N_COMPONENT; ++i)
         d_img[id * N_COMPONENT + i] = FULL - d_img[id * N_COMPONENT + i];
   }
}

__global__ void sobel_kernel(unsigned char *d_img, const unsigned char *d_tmp, const int height,
                             const int width)
{
   int id = index();

   int size = height * width;

   if (id < size) {
      for (size_t i = 0; i < N_COMPONENT; ++i) {
         // Convolution matrices for computing approximations of horizontal and vertical
         // derivatives
         int gx = 0;
         int gy = 0;

         // Horizontal approximation
         if (id - width - 1 > 0)
            gx -= 1 * (int)d_tmp[(id - width - 1) * N_COMPONENT + i];
         if (id - 1 > 0)
            gx -= 2 * (int)d_tmp[(id - 1) * N_COMPONENT + i];
         if (id + width - 1 < size)
            gx -= 1 * (int)d_tmp[(id + width - 1) * N_COMPONENT + i];

         if (id - width + 1 > 0)
            gx += 1 * (int)d_tmp[(id - width + 1) * N_COMPONENT + i];
         if (id + 1 < size)
            gx += 2 * (int)d_tmp[(id + 1) * N_COMPONENT + i];
         if (id + width + 1 < size)
            gx += 1 * (int)d_tmp[(id + width + 1) * N_COMPONENT + i];

         // Vertical approximation
         if (id + width - 1 < size)
            gy -= 1 * (int)d_tmp[(id + width - 1) * N_COMPONENT + i];
         if (id + width < size)
            gy -= 2 * (int)d_tmp[(id + width) * N_COMPONENT + i];
         if (id + width + 1 < size)
            gy -= 1 * (int)d_tmp[(id + width + 1) * N_COMPONENT + i];

         if (id - width - 1 > 0)
            gy += 1 * (int)d_tmp[(id - width - 1) * N_COMPONENT + i];
         if (id - width > 0)
            gy += 2 * (int)d_tmp[(id - width) * N_COMPONENT + i];
         if (id - width + 1 > 0)
            gy += 1 * (int)d_tmp[(id - width + 1) * N_COMPONENT + i];

         //
         d_img[id * N_COMPONENT + i] = (unsigned char)sqrt((float)(gx * gx + gy * gy));
      }
   }
}

__global__ void resize_kernel(unsigned char *d_img, const unsigned char *d_tmp, const size_t old_w,
                              const size_t old_h, const size_t new_w, const size_t new_h,
                              const int off_x, const int off_y)
{
   size_t col = blockIdx.x * blockDim.x + threadIdx.x;
   size_t row = blockIdx.y * blockDim.y + threadIdx.y;

   if (row + off_x < new_w && row + off_x > 0 && col + off_y < new_h && col + off_y > 0) {
      const float w_factor = (float)new_w / (float)old_w;
      const float h_factor = (float)new_h / (float)old_h;

      const size_t old_row = (size_t)((float)row / w_factor);
      const size_t old_col = (size_t)((float)col / h_factor);

      const size_t old_idx = (old_col * old_w + old_row + off_y) * N_COMPONENT;
      const size_t new_idx = ((col + off_x) * old_w + row + off_y) * N_COMPONENT;

      for (size_t i = 0; i < N_COMPONENT; ++i)
         d_img[new_idx + i] = d_tmp[old_idx + i];
   }
}
