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

#endif // _KERNEL_H_
