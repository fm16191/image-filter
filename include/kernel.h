#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define gpuErrCheck(ans)                                                                           \
   {                                                                                               \
      gpuAssert((ans), __FILE__, __LINE__);                                                        \
   }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
      fprintf(stderr, "GPUassert: %s | %s:%d\n", cudaGetErrorString(code), file, line);
}

__global__ void saturate_component(unsigned char *d_img, const size_t size, const size_t index);
__global__ void horizontal_flip_kernel(unsigned char *d_img, const unsigned char *d_tmp,
                                       const size_t width, const size_t size);
__global__ void vertical_flip_kernel(unsigned char *d_img, const unsigned char *d_tmp,
                                     const size_t size);
__global__ void blur_kernel(unsigned char *d_img, const unsigned char *d_tmp, const size_t height,
                            const size_t width);
__global__ void grayscale_kernel(unsigned char *d_img, const size_t size);
__global__ void sobel_kernel(unsigned char *d_img, const unsigned char *d_tmp, const int height,
                             const int width);
__global__ void negative_kernel(unsigned char *d_img, const size_t size);
__global__ void extract_component(unsigned char *d_img, const size_t size,
                                  const size_t component_index);
__global__ void resize_kernel(unsigned char *d_img, const unsigned char *d_tmp, const size_t old_w,
                              const size_t old_h, const size_t new_w, const size_t new_h,
                              const int off_x, const int off_y);

#endif // _KERNEL_H_
