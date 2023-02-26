#include "FreeImage.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 1920
#define HEIGHT 1024
#define BPP 24        // Since we're outputting three 8 bit RGB values
#define N_THREADS 32  // Number of CUDA threads
#define N_COMPONENT 3 // we have 3 component (RGB)
#define FULL 255      // Max rgb value
#define gpuErrCheck(ans)                                                                           \
   {                                                                                               \
      gpuAssert((ans), __FILE__, __LINE__);                                                        \
   }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
      fprintf(stderr, "GPUassert: %s | %s:%d\n", cudaGetErrorString(code), file, line);
}

__global__ void saturate_component(unsigned char *d_img, const size_t size, const size_t index)
{
   size_t block_size = blockDim.x * blockDim.y;
   size_t block_id = gridDim.x * blockIdx.y + blockIdx.x;
   size_t thread_id = blockDim.x * threadIdx.y + threadIdx.x;
   size_t id = block_id * block_size + thread_id;

   if (id < size)
      d_img[id * N_COMPONENT + index] = FULL;
}

__host__ int usage(char *exec)
{
   printf("Usage : %s -i <in.jpg> -o <out.jpg> [-s]\n", exec);

   printf("\n");

   printf("Options : \n"
          "-i, --input <in.jpg>    Input JPG filepath. Default : `img.jpg`\n"
          "-o, --output <out.jpg>  Output JPG filepath. Default : `new_img.jpg`\n"

          "-s, --saturate <r,g,b>  Saturate an RGB component of the image\n"

          "-h, --help              Show this message and exit\n"
          // "-d, --debug          Enable debug mode\n"
   );
   return 0;
}

__host__ int hasarg(size_t i, int argc, char **argv)
{
   if (i + 1 >= (size_t)argc || argv[i + 1][0] == '-') {
      printf("Missing variable\n");
      usage(argv[0]);
      exit(EXIT_FAILURE);
   }
   else
      return 1;
}

enum saturate_t {
   R,
   G,
   B,
   NOSATURATION
};

__host__ void saturate_image(dim3 dim_grid, dim3 dim_block, unsigned char *d_img, size_t height,
                             size_t width, saturate_t saturate)
{
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   cudaEventRecord(start);
   saturate_component<<<dim_grid, dim_block>>>(d_img, height * width, saturate);
   cudaEventRecord(stop);

   cudaDeviceSynchronize();
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Image saturation (%s) in %e s\n",
          (saturate == R) ? "red" : (saturate == G ? "green" : "blue"), milliseconds / 1e3);
}

int main(int argc, char **argv)
{
   size_t i = 1;
   size_t debug = 0;
   cudaError_t err;

   char *input = strdup("img.jpg");
   char *output = strdup("new_img.jpg");

   enum saturate_t saturate = NOSATURATION;

   while (i < (size_t)argc && strlen(argv[i]) > 1 && argv[i][0] == '-') {
      if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
         return usage(argv[0]);
      else if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--debug")) {
         printf("Since debug mode has been activated, repetitions are set to 1.\n");
         debug = 1;
      }
      else if (!strcmp(argv[i], "-i") || !strcmp(argv[i], "--input")) {
         if (hasarg(i, argc, argv))
            input = argv[i + 1];
         i++;
      }
      else if (!strcmp(argv[i], "-o") || !strcmp(argv[i], "--output")) {
         if (hasarg(i, argc, argv))
            output = argv[i + 1];
         i++;
      }
      else if (!strcmp(argv[i], "-s") || !strcmp(argv[i], "--saturate")) {
         if (hasarg(i, argc, argv)) {
            if (!strcmp(argv[i + 1], "r"))
               saturate = R;
            else if (!strcmp(argv[i + 1], "g"))
               saturate = G;
            else if (!strcmp(argv[i + 1], "b"))
               saturate = B;
            else
               return printf("--saturate option must be in <r,g,b>\n"), usage(argv[0]);
         }
         i++;
      }
      i++;
   }

   FreeImage_Initialise();

   // load and decode a regular file
   FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(input);
   FIBITMAP *bitmap = FreeImage_Load(FIF_JPEG, input, 0);

   if (!bitmap)
      return fprintf(stderr, "Cannot load Image\n"), 1;

   unsigned width = FreeImage_GetWidth(bitmap);
   unsigned height = FreeImage_GetHeight(bitmap);
   unsigned pitch = FreeImage_GetPitch(bitmap);

   fprintf(stderr, "Processing Image of size %d x %d\n", width, height);

   // Allocate memories
   unsigned int size_in_bytes = sizeof(unsigned char) * N_COMPONENT * width * height;
   unsigned char *h_img = NULL;
   unsigned char *d_img = NULL;
   unsigned char *d_tmp = NULL;

   h_img = (unsigned char *)malloc(size_in_bytes);
   if (!h_img)
      return fprintf(stderr, "Cannot allocate memory\n"), 2;
   err = cudaMalloc(&d_img, size_in_bytes);
   gpuErrCheck(err);
   err = cudaMalloc(&d_tmp, size_in_bytes);
   gpuErrCheck(err);

   // Get pixels
   BYTE *bits = (BYTE *)FreeImage_GetBits(bitmap);
   for (int y = 0; y < height; y++) {
      BYTE *pixel = (BYTE *)bits;
      for (int x = 0; x < width; x++) {
         int idx = ((y * width) + x) * N_COMPONENT;
         h_img[idx + 0] = pixel[FI_RGBA_RED];
         h_img[idx + 1] = pixel[FI_RGBA_GREEN];
         h_img[idx + 2] = pixel[FI_RGBA_BLUE];
         pixel += N_COMPONENT;
      }
      // next line
      bits += pitch;
   }

   // Copy host array to device array
   err = cudaMemcpy(d_img, h_img, size_in_bytes, cudaMemcpyHostToDevice);
   gpuErrCheck(err);

   // Define grid and blocks
   size_t grid_x = width / N_THREADS + 1;
   size_t grid_y = height / N_THREADS + 1;
   dim3 dim_grid(grid_x, grid_y);
   dim3 dim_block(N_THREADS, N_THREADS);

   fprintf(stderr, "Using a grid (%d, %d, %d) of blocks (%d, %d, %d)\n", dim_grid.x, dim_grid.y,
           dim_grid.z, dim_block.x, dim_block.y, dim_block.z);

   //
   if (saturate != NOSATURATION)
      saturate_image(dim_grid, dim_block, d_img, height, width, saturate);

   // Kernel
   // for (int y = 0; y < height; y++) {
   //    for (int x = 0; x < width; x++) {
   //       int ida = ((y * width) + x) * 3;
   //       int idb = ((width * height) - ((y * width) + x)) * 3;
   //       d_img[ida + 0] = d_tmp[idb + 0];
   //       d_img[ida + 1] = d_tmp[idb + 1];
   //       d_img[ida + 2] = d_tmp[idb + 2];
   //    }
   // }

   // for (int y = 0; y < height / 2; y++) {
   //    for (int x = 0; x < width / 2; x++) {
   //       // if( x < (width/2 * 0.75) || y < (height/2 * 0.65))
   //       {
   //          int idx = ((y * width) + x) * 3;
   //          d_img[idx + 0] /= 2;
   //          d_img[idx + 1] /= 4;
   //          d_img[idx + 2] = 0xFF / 1.5;
   //       }
   //    }
   // }

   // for (int y = height / 2; y < height; y++) {
   //    for (int x = width / 2; x < width; x++) {
   //       // if( x >= ((width/2) + (width/2 * 0.25)) || y >= ((height/2) + (height/2 * 0.35)))
   //       {
   //          int idx = ((y * width) + x) * 3;
   //          d_img[idx + 0] = 0xFF - d_img[idx + 0];
   //          d_img[idx + 1] = 0xFF / 2;
   //          d_img[idx + 2] /= 4;
   //       }
   //    }
   // }

   // for (int y = height / 2; y < height; y++) {
   //    for (int x = 0; x < width / 2; x++) {
   //       // if( x < (width/2 * 0.75) || y >= (height/2) + (height/2 * 0.35))
   //       {
   //          int idx = ((y * width) + x) * 3;
   //          d_img[idx + 0] = 0xFF / 2;
   //          d_img[idx + 1] /= 2;
   //          d_img[idx + 2] /= 2;
   //       }
   //    }
   // }

   // for (int y = 0; y < height / 2; y++) {
   //    for (int x = width / 2; x < width; x++) {
   //       // if( x >= ((width/2) + (width/2 * 0.25)) || y < (height/2 * 0.65))
   //       {
   //          int idx = ((y * width) + x) * 3;
   //          int grey = d_img[idx + 0] * 0.299 + d_img[idx + 1] * 0.587 + d_img[idx + 2] * 0.114;
   //          // d_img[idx + 0] = 0xFF - d_img[idx + 0];
   //          // d_img[idx + 1] = 0xFF - d_img[idx + 1];
   //          // d_img[idx + 2] = 0xFF - d_img[idx + 2];
   //          d_img[idx + 0] = grey;
   //          d_img[idx + 1] = grey;
   //          d_img[idx + 2] = grey;
   //       }
   //    }
   // }

   // // Copy back
   // memcpy(img, d_img, 3 * width * height * sizeof(unsigned int));
   err = cudaMemcpy(h_img, d_img, size_in_bytes, cudaMemcpyDeviceToHost);
   gpuErrCheck(err);

   // Store pixels
   bits = (BYTE *)FreeImage_GetBits(bitmap);
   for (int y = 0; y < height; y++) {
      BYTE *pixel = (BYTE *)bits;
      for (int x = 0; x < width; x++) {
         RGBQUAD newcolor;

         int idx = (y * width + x) * N_COMPONENT;
         newcolor.rgbRed = h_img[idx + 0];
         newcolor.rgbGreen = h_img[idx + 1];
         newcolor.rgbBlue = h_img[idx + 2];

         if (!FreeImage_SetPixelColor(bitmap, x, y, &newcolor))
            fprintf(stderr, "(%d, %d) Fail...\n", x, y);

         pixel += N_COMPONENT;
      }
      // next line
      bits += pitch;
   }

   if (FreeImage_Save(FIF_PNG, bitmap, output, 0))
      printf("Image successfully saved ! \n");

   FreeImage_DeInitialise(); // Cleanup !

   free(h_img);
   cudaFree(d_img);
   cudaFree(d_tmp);
}
