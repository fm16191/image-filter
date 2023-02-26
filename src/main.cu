#include "FreeImage.h"
#include "kernel.h"
#include "utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__host__ static void load_pixels(FIBITMAP *bitmap, unsigned char *h_img, const size_t height,
                                 const size_t width, const size_t pitch)
{
   BYTE *bits = (BYTE *)FreeImage_GetBits(bitmap);
   for (size_t y = 0; y < height; y++) {
      BYTE *pixel = (BYTE *)bits;
      for (size_t x = 0; x < width; x++) {
         size_t idx = ((y * width) + x) * N_COMPONENT;
         h_img[idx + 0] = pixel[FI_RGBA_RED];
         h_img[idx + 1] = pixel[FI_RGBA_GREEN];
         h_img[idx + 2] = pixel[FI_RGBA_BLUE];
         pixel += N_COMPONENT;
      }
      bits += pitch; // next line
   }
}

__host__ static void store_pixels(FIBITMAP *bitmap, unsigned char *h_img, const size_t height,
                                  const size_t width, const size_t pitch)
{
   BYTE *bits = (BYTE *)FreeImage_GetBits(bitmap);
   for (size_t y = 0; y < height; y++) {
      BYTE *pixel = (BYTE *)bits;
      for (size_t x = 0; x < width; x++) {
         RGBQUAD newcolor;

         size_t idx = (y * width + x) * N_COMPONENT;
         newcolor.rgbRed = h_img[idx + 0];
         newcolor.rgbGreen = h_img[idx + 1];
         newcolor.rgbBlue = h_img[idx + 2];

         if (!FreeImage_SetPixelColor(bitmap, x, y, &newcolor))
            fprintf(stderr, "Cannot set (%ld, %ld) pixel color\n", x, y);

         pixel += N_COMPONENT;
      }
      bits += pitch; // next line
   }
}

__host__ int usage(char *exec)
{
   printf("Usage : %s -i <in.jpg> -o <out.jpg> [-s]\n", exec);

   printf("\n");

   printf("Options : \n"
          "-i, --input <in.jpg>    Input JPG filepath. Default : `img.jpg`\n"
          "-o, --output <out.jpg>  Output JPG filepath. Default : `new_img.jpg`\n"

          "-s, --saturate <r,g,b>  Saturate an RGB component of the image\n"
          "-f, --flip <h,v>        Flip image horizontally, vertically\n"

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

__host__ void flip_image(dim3 dim_grid, dim3 dim_block, unsigned char *d_img, unsigned char *d_tmp,
                         size_t height, size_t width, orientation_t orientation)
{
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);

   cudaEventRecord(start);
   if (orientation == HORIZONTAL)
      horizontal_flip_kernel<<<dim_grid, dim_block>>>(d_img, d_tmp, width, height * width);
   else
      vertical_flip_kernel<<<dim_grid, dim_block>>>(d_img, d_tmp, height * width);
   cudaEventRecord(stop);

   cudaDeviceSynchronize();
   cudaEventSynchronize(stop);
   float milliseconds = 0;
   cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Image flip (%s) in %e s\n", (orientation == HORIZONTAL) ? "horizontally" : "vertically",
          milliseconds / 1e3);
}

int main(int argc, char **argv)
{
   size_t i = 1;
   size_t debug = 0;
   cudaError_t err;

   char *input = strdup("img.jpg");
   char *output = strdup("new_img.jpg");

   enum saturate_t saturate = NOSATURATION;
   enum orientation_t orientation = NOFLIP;

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
      else if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--flip")) {
         if (hasarg(i, argc, argv)) {
            if (!strcmp(argv[i + 1], "h"))
               orientation = HORIZONTAL;
            else if (!strcmp(argv[i + 1], "v"))
               orientation = VERTICAL;
            else
               return printf("--flip option must be in <h,v>\n"), usage(argv[0]);
         }
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

   h_img = (unsigned char *)malloc(size_in_bytes);
   if (!h_img)
      return fprintf(stderr, "Cannot allocate memory\n"), 2;
   err = cudaMalloc(&d_img, size_in_bytes);
   gpuErrCheck(err);

   // Get pixels
   load_pixels(bitmap, h_img, height, width, pitch);

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

   if (orientation != NOFLIP) {
      unsigned char *d_tmp = NULL;
      err = cudaMalloc(&d_tmp, size_in_bytes);
      gpuErrCheck(err);

      err = cudaMemcpy(d_tmp, h_img, size_in_bytes, cudaMemcpyHostToDevice);
      gpuErrCheck(err);

      flip_image(dim_grid, dim_block, d_img, d_tmp, height, width, orientation);

      cudaFree(d_tmp);
   }

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
   store_pixels(bitmap, h_img, height, width, pitch);

   if (FreeImage_Save(FIF_PNG, bitmap, output, 0))
      printf("Image successfully saved ! \n");

   FreeImage_DeInitialise(); // Cleanup !

   free(h_img);
   cudaFree(d_img);
   // cudaFree(d_tmp);
}
