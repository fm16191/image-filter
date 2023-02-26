// #include "FreeImage.h"
// #include "utils.h"

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <stdio.h>

// __host__ static void load_pixels(FIBITMAP *bitmap, unsigned char *h_img, const size_t height,
//                                  const size_t width, const size_t pitch)
// {
//    BYTE *bits = (BYTE *)FreeImage_GetBits(bitmap);
//    for (size_t y = 0; y < height; y++) {
//       BYTE *pixel = (BYTE *)bits;
//       for (size_t x = 0; x < width; x++) {
//          size_t idx = ((y * width) + x) * N_COMPONENT;
//          h_img[idx + 0] = pixel[FI_RGBA_RED];
//          h_img[idx + 1] = pixel[FI_RGBA_GREEN];
//          h_img[idx + 2] = pixel[FI_RGBA_BLUE];
//          pixel += N_COMPONENT;
//       }
//       bits += pitch; // next line
//    }
// }

// __host__ static void store_pixels(FIBITMAP *bitmap, unsigned char *h_img, const size_t height,
//                                   const size_t width, const size_t pitch)
// {
//    BYTE *bits = (BYTE *)FreeImage_GetBits(bitmap);
//    for (size_t y = 0; y < height; y++) {
//       BYTE *pixel = (BYTE *)bits;
//       for (size_t x = 0; x < width; x++) {
//          RGBQUAD newcolor;

//          size_t idx = (y * width + x) * N_COMPONENT;
//          newcolor.rgbRed = h_img[idx + 0];
//          newcolor.rgbGreen = h_img[idx + 1];
//          newcolor.rgbBlue = h_img[idx + 2];

//          if (!FreeImage_SetPixelColor(bitmap, x, y, &newcolor))
//             fprintf(stderr, "Cannot set (%ld, %ld) pixel color\n", x, y);

//          pixel += N_COMPONENT;
//       }
//       bits += pitch; // next line
//    }
// }

// __host__ int usage(char *exec)
// {
//    printf("Usage : %s -i <in.jpg> -o <out.jpg> [-s]\n", exec);

//    printf("\n");

//    printf("Options : \n"
//           "-i, --input <in.jpg>    Input JPG filepath. Default : `img.jpg`\n"
//           "-o, --output <out.jpg>  Output JPG filepath. Default : `new_img.jpg`\n"

//           "-s, --saturate <r,g,b>  Saturate an RGB component of the image\n"

//           "-h, --help              Show this message and exit\n"
//           // "-d, --debug          Enable debug mode\n"
//    );
//    return 0;
// }

// __host__ int hasarg(size_t i, int argc, char **argv)
// {
//    if (i + 1 >= (size_t)argc || argv[i + 1][0] == '-') {
//       printf("Missing variable\n");
//       usage(argv[0]);
//       exit(EXIT_FAILURE);
//    }
//    else
//       return 1;
// }