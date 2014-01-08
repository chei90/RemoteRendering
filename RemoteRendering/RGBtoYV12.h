#ifndef RGBTOYV12_H_
#define RGBTOYV12_H_
#include <Windows.h>
#include <cuda.h>


extern void callKernel(int width, int height, unsigned char* yuv,  unsigned char* pixels);


#endif