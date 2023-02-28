# CUDA Image filtering library

*image-filter* is a set of CUDA kernels that filters an image using CUDA kernels.

## Dependances
To run, *image-filter* requires :
- FreeImage library installed and in PATH
- a CUDA-capable GPU

## Installation
```bash
$ make
```

## Usage
```bash
$ ./filter -h

Usage : ./filter -i <in.jpg> -o <out.jpg> [-hglnp] [-s <r,g,b>] [-x <r,g,b>] [-f <h,v>] [-b [it]] [-r WxH[+x+y]]

Options : 
-i, --input <in.jpg>    Input JPG filepath. Default : `img.jpg`
-o, --output <out.jpg>  Output JPG filepath. Default : `new_img.jpg`

-s, --saturate <r,g,b>  Saturate an RGB component of the image
-x, --extract <r,g,b>   Extract an RGB component of the image
-f, --flip <h,v>        Flip image horizontally, vertically
-b, --blur [it]         Blur image `it` times. Default : `1`
-g, --grayscale         Gray scale image.
-l, --sobel             Apply a Sobel filter to the image.
-n, --negative          Transform image into a negative.
-r, --resize WxH[+x+y]  Resize image with WxH dimensions at x,y offsets.
-p, --pop-art           Create a pop-art combinaison out of the original image.

-h, --help              Show this message and exit
