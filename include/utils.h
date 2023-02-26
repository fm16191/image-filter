#ifndef _UTILS_H_
#define _UTILS_H_

// #include "FreeImage.h"

#define WIDTH 1920
#define HEIGHT 1024
#define BPP 24        // Since we're outputting three 8 bit RGB values
#define N_THREADS 32  // Number of CUDA threads
#define N_COMPONENT 3 // we have 3 component (RGB)
#define FULL 255      // Max rgb value

enum saturate_t {
   R,
   G,
   B,
   NOSATURATION
};

// __host__ static void load_pixels(FIBITMAP *bitmap, unsigned char *h_img, const size_t height,
//                                  const size_t width, const size_t pitch);
// __host__ static void store_pixels(FIBITMAP *bitmap, unsigned char *h_img, const size_t height,
//                                   const size_t width, const size_t pitch);

// __host__ int usage(char *exec);
// __host__ int hasarg(size_t i, int argc, char **argv);

#endif