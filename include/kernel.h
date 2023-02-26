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

#endif // _KERNEL_H_
