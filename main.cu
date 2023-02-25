#include "FreeImage.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 1920
#define HEIGHT 1024
#define BPP 24 // Since we're outputting three 8 bit RGB values
#define FILENAME_BUFFER_LEN 1024

using namespace std;

int usage(char *exec)
{
   printf("Usage : %s -i <in.png> -o <out.png> [-s]\n", exec);

   printf("\n");

   printf("Options : \n"
          "-i, --input <in.png>    Input JPG filepath. Default : `img.jpg`\n"
          "-o, --output <out.png>  Output JPG filepath. Default : `new_img.jpg`\n"

          "-s, --saturate <r,g,b>  Saturate an RGB component of the image\n"

          "-h, --help              Show this message and exit\n"
          // "-d, --debug          Enable debug mode\n"
   );
   return 0;
}

int hasarg(size_t i, int argc, char **argv)
{
   if (i + 1 >= (size_t)argc || argv[i + 1][0] == '-') {
      printf("Missing variable\n");
      usage(argv[0]);
      exit(EXIT_FAILURE);
   }
   else
      return 1;
}

int main(int argc, char **argv)
{
   size_t i = 1;
   size_t debug = 0;
   // char input[FILENAME_BUFFER_LEN] = "img.jpg";
   // char output[FILENAME_BUFFER_LEN] = "new_img.jpg";
   std::string input = "img.jpg";
   std::string output = "new_img.jpg";

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
      i++;
   }

   FreeImage_Initialise();
   const char *PathName = "img.jpg";
   const char *PathDest = "new_img.png";
   // load and decode a regular file
   FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(PathName);

   FIBITMAP *bitmap = FreeImage_Load(FIF_JPEG, PathName, 0);

   if (!bitmap)
      exit(1); // WTF?! We can't even allocate images ? Die !

   unsigned width = FreeImage_GetWidth(bitmap);
   unsigned height = FreeImage_GetHeight(bitmap);
   unsigned pitch = FreeImage_GetPitch(bitmap);

   fprintf(stderr, "Processing Image of size %d x %d\n", width, height);

   unsigned int *img = (unsigned int *)malloc(sizeof(unsigned int) * 3 * width * height);
   unsigned int *d_img = (unsigned int *)malloc(sizeof(unsigned int) * 3 * width * height);
   unsigned int *d_tmp = (unsigned int *)malloc(sizeof(unsigned int) * 3 * width * height);

   BYTE *bits = (BYTE *)FreeImage_GetBits(bitmap);
   for (int y = 0; y < height; y++) {
      BYTE *pixel = (BYTE *)bits;
      for (int x = 0; x < width; x++) {
         int idx = ((y * width) + x) * 3;
         img[idx + 0] = pixel[FI_RGBA_RED];
         img[idx + 1] = pixel[FI_RGBA_GREEN];
         img[idx + 2] = pixel[FI_RGBA_BLUE];
         pixel += 3;
      }
      // next line
      bits += pitch;
   }

   memcpy(d_img, img, 3 * width * height * sizeof(unsigned int));
   memcpy(d_tmp, img, 3 * width * height * sizeof(unsigned int));

   // // Kernel
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

   // Copy back
   memcpy(img, d_img, 3 * width * height * sizeof(unsigned int));

   bits = (BYTE *)FreeImage_GetBits(bitmap);
   for (int y = 0; y < height; y++) {
      BYTE *pixel = (BYTE *)bits;
      for (int x = 0; x < width; x++) {
         RGBQUAD newcolor;

         int idx = ((y * width) + x) * 3;
         newcolor.rgbRed = img[idx + 0];
         newcolor.rgbGreen = img[idx + 1];
         newcolor.rgbBlue = img[idx + 2];

         if (!FreeImage_SetPixelColor(bitmap, x, y, &newcolor)) {
            fprintf(stderr, "(%d, %d) Fail...\n", x, y);
         }

         pixel += 3;
      }
      // next line
      bits += pitch;
   }

   if (FreeImage_Save(FIF_PNG, bitmap, PathDest, 0))
      cout << "Image successfully saved ! " << endl;
   FreeImage_DeInitialise(); // Cleanup !

   free(img);
   free(d_img);
   free(d_tmp);
}
