#include "FreeImage.h"
#include "kernel.h"
#include "utils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ALLOC_SIZE_BYTES (sizeof(unsigned char) * N_COMPONENT * width * height)

static cudaEvent_t start, stop;

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

__host__ float cudaTimerCompute(cudaEvent_t start, cudaEvent_t stop)
{
   cudaDeviceSynchronize();
   cudaEventSynchronize(stop);
   float milliseconds = 0.;
   cudaEventElapsedTime(&milliseconds, start, stop);
   return milliseconds;
}

__host__ int usage(char *exec)
{
   printf("Usage : %s -i <in.jpg> -o <out.jpg> [-s]\n", exec);

   printf("\n");

   printf("Options : \n"
          "-i, --input <in.jpg>    Input JPG filepath. Default : `img.jpg`\n"
          "-o, --output <out.jpg>  Output JPG filepath. Default : `new_img.jpg`\n"

          "-s, --saturate <r,g,b>  Saturate an RGB component of the image\n"
          "-s, --extract <r,g,b>   Extract an RGB component of the image\n"
          "-f, --flip <h,v>        Flip image horizontally, vertically\n"
          "-b, --blur [it]         Blur image `it` times. Default : `1`\n"
          "-g, --grayscale         Gray scale image.\n"
          "-l, --sobel             Apply a Sobel filter to the image.\n"
          "-n, --negative          Transform image into a negative.\n"

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
                             size_t width, component_t component)
{
   cudaEventRecord(start);
   saturate_component<<<dim_grid, dim_block>>>(d_img, height * width, component);
   cudaEventRecord(stop);

   float milliseconds = cudaTimerCompute(start, stop);
   printf("Image saturation (%s) in %e s\n",
          (component == R) ? "red" : (component == G ? "green" : "blue"), milliseconds / 1e3);
}

__host__ void color_extraction_image(dim3 dim_grid, dim3 dim_block, unsigned char *d_img,
                                     size_t height, size_t width, component_t component)
{
   cudaEventRecord(start);
   extract_component<<<dim_grid, dim_block>>>(d_img, height * width, component);
   cudaEventRecord(stop);

   float milliseconds = cudaTimerCompute(start, stop);
   printf("Image color extraction (%s) in %e s\n",
          (component == R) ? "red" : (component == G ? "green" : "blue"), milliseconds / 1e3);
}

__host__ void flip_image(dim3 dim_grid, dim3 dim_block, unsigned char *d_img, unsigned char *d_tmp,
                         size_t height, size_t width, orientation_t orientation)
{
   cudaError_t err = cudaMemcpy(d_tmp, d_img, ALLOC_SIZE_BYTES, cudaMemcpyDeviceToDevice);
   gpuErrCheck(err);

   cudaEventRecord(start);
   if (orientation == HORIZONTAL)
      horizontal_flip_kernel<<<dim_grid, dim_block>>>(d_img, d_tmp, width, height * width);
   else
      vertical_flip_kernel<<<dim_grid, dim_block>>>(d_img, d_tmp, height * width);
   cudaEventRecord(stop);

   float milliseconds = cudaTimerCompute(start, stop);
   printf("Image flip (%s) in %e s\n", (orientation == HORIZONTAL) ? "horizontally" : "vertically",
          milliseconds / 1e3);
}

__host__ void blur_image(dim3 dim_grid, dim3 dim_block, unsigned char *d_img, unsigned char *d_tmp,
                         size_t height, size_t width)
{
   cudaError_t err = cudaMemcpy(d_tmp, d_img, ALLOC_SIZE_BYTES, cudaMemcpyDeviceToDevice);
   gpuErrCheck(err);

   cudaEventRecord(start);
   blur_kernel<<<dim_grid, dim_block>>>(d_img, d_tmp, height, width);
   cudaEventRecord(stop);

   float milliseconds = cudaTimerCompute(start, stop);
   printf("Image blurred in %e s\n", milliseconds / 1e3);
}

__host__ void grayscale_image(dim3 dim_grid, dim3 dim_block, unsigned char *d_img, size_t height,
                              size_t width)
{
   cudaEventRecord(start);
   grayscale_kernel<<<dim_grid, dim_block>>>(d_img, height * width);
   cudaEventRecord(stop);

   float milliseconds = cudaTimerCompute(start, stop);
   printf("Image grayscaled in %e s\n", milliseconds / 1e3);
}

__host__ void negative_image(dim3 dim_grid, dim3 dim_block, unsigned char *d_img, size_t height,
                             size_t width)
{
   cudaEventRecord(start);
   negative_kernel<<<dim_grid, dim_block>>>(d_img, height * width);
   cudaEventRecord(stop);

   float milliseconds = cudaTimerCompute(start, stop);
   printf("Image negative in %e s\n", milliseconds / 1e3);
}

__host__ void sobel_image(dim3 dim_grid, dim3 dim_block, unsigned char *d_img, unsigned char *d_tmp,
                          size_t height, size_t width)
{
   cudaError_t err = cudaMemcpy(d_tmp, d_img, ALLOC_SIZE_BYTES, cudaMemcpyDeviceToDevice);
   gpuErrCheck(err);

   cudaEventRecord(start);
   sobel_kernel<<<dim_grid, dim_block>>>(d_img, d_tmp, height, width);
   cudaEventRecord(stop);

   float milliseconds = cudaTimerCompute(start, stop);
   printf("Image filtered with sobel in %e s\n", milliseconds / 1e3);
}

int main(int argc, char **argv)
{
   size_t i = 1;
   // size_t debug = 0;
   cudaError_t err;

   char *input = strdup("img.jpg");
   char *output = strdup("new_img.jpg");

   enum component_t component = NO_COMPONENT;
   enum orientation_t orientation = NO_ORIENTATION;

   /* Parse program options */
   while (i < (size_t)argc && strlen(argv[i]) > 1 && argv[i][0] == '-') {
      if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help"))
         return usage(argv[0]);
      // else if (!strcmp(argv[i], "-d") || !strcmp(argv[i], "--debug")) {
      //    printf("Since debug mode has been activated, repetitions are set to 1.\n");
      //    debug = 1;
      // }
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
      i++;
   }

   /* -------------- */
   /* Initialisation */
   /* -------------- */

   cudaEventCreate(&start);
   cudaEventCreate(&stop);

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
   unsigned char *h_img = NULL;
   unsigned char *d_img = NULL;
   unsigned char *d_tmp = NULL;

   h_img = (unsigned char *)malloc(ALLOC_SIZE_BYTES);
   if (!h_img)
      return fprintf(stderr, "Cannot allocate memory\n"), 2;
   err = cudaMalloc(&d_img, ALLOC_SIZE_BYTES);
   gpuErrCheck(err);
   err = cudaMalloc(&d_tmp, ALLOC_SIZE_BYTES);
   gpuErrCheck(err);

   // Get pixels
   load_pixels(bitmap, h_img, height, width, pitch);

   // Copy host array to device array
   err = cudaMemcpy(d_img, h_img, ALLOC_SIZE_BYTES, cudaMemcpyHostToDevice);
   gpuErrCheck(err);

   // Define grid and blocks
   size_t grid_x = width / N_THREADS + 1;
   size_t grid_y = height / N_THREADS + 1;
   dim3 dim_grid(grid_x, grid_y);
   dim3 dim_block(N_THREADS, N_THREADS);

   fprintf(stderr, "Using a grid (%d, %d, %d) of blocks (%d, %d, %d)\n", dim_grid.x, dim_grid.y,
           dim_grid.z, dim_block.x, dim_block.y, dim_block.z);

   /* Parse filters arguments */
   i = 1;
   while (i < (size_t)argc && strlen(argv[i]) > 1 && argv[i][0] == '-') {
      if (!strcmp(argv[i], "-b") || !strcmp(argv[i], "--blur")) {
         size_t max_it = 1;
         if (hasarg(i, argc, argv)) {
            max_it = atoi(argv[i + 1]);
            i++;
         }
         for (size_t it = 0; it < max_it; ++it)
            blur_image(dim_grid, dim_block, d_img, d_tmp, height, width);
      }
      else if (!strcmp(argv[i], "-g") || !strcmp(argv[i], "--grayscale")) {
         grayscale_image(dim_grid, dim_block, d_img, height, width);
      }
      else if (!strcmp(argv[i], "-n") || !strcmp(argv[i], "--negative")) {
         negative_image(dim_grid, dim_block, d_img, height, width);
      }
      else if (!strcmp(argv[i], "-l") || !strcmp(argv[i], "--sobel")) {
         sobel_image(dim_grid, dim_block, d_img, d_tmp, height, width);
      }
      else if (!strcmp(argv[i], "-f") || !strcmp(argv[i], "--flip")) {
         if (hasarg(i, argc, argv)) {
            if (!strcmp(argv[i + 1], "h"))
               orientation = HORIZONTAL;
            else if (!strcmp(argv[i + 1], "v"))
               orientation = VERTICAL;
            else
               return printf("--flip option must be in <h,v>\n"), usage(argv[0]);
            //
            flip_image(dim_grid, dim_block, d_img, d_tmp, height, width, orientation);
         }
         i++;
      }
      else if (!strcmp(argv[i], "-s") || !strcmp(argv[i], "--saturate")) {
         if (hasarg(i, argc, argv)) {
            if (!strcmp(argv[i + 1], "r"))
               component = R;
            else if (!strcmp(argv[i + 1], "g"))
               component = G;
            else if (!strcmp(argv[i + 1], "b"))
               component = B;
            else
               return printf("--saturate option must be in <r,g,b>\n"), usage(argv[0]);
            //
            saturate_image(dim_grid, dim_block, d_img, height, width, component);
         }
         i++;
      }
      else if (!strcmp(argv[i], "-x") || !strcmp(argv[i], "--extract")) {
         if (hasarg(i, argc, argv)) {
            if (!strcmp(argv[i + 1], "r"))
               component = R;
            else if (!strcmp(argv[i + 1], "g"))
               component = G;
            else if (!strcmp(argv[i + 1], "b"))
               component = B;
            else
               return printf("--extract option must be in <r,g,b>\n"), usage(argv[0]);
            //
            color_extraction_image(dim_grid, dim_block, d_img, height, width, component);
         }
         i++;
      }
      i++;
   }

   /* ----------- */
   /* Termination */
   /* ----------- */
   // Copy back
   err = cudaMemcpy(h_img, d_img, ALLOC_SIZE_BYTES, cudaMemcpyDeviceToHost);
   gpuErrCheck(err);

   // Store pixels
   store_pixels(bitmap, h_img, height, width, pitch);

   if (FreeImage_Save(FIF_PNG, bitmap, output, 0))
      printf("Image successfully saved ! \n");

   FreeImage_DeInitialise(); // Cleanup !

   free(h_img);
   cudaFree(d_img);
   cudaFree(d_tmp);
}
