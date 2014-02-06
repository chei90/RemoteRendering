#ifndef RGBTOYV12_H_
#define RGBTOYV12_H_
#include <Windows.h>
#include <cuda.h>
#include <cuda_runtime.h>


extern void callKernelD3D(int width, int height, unsigned char* yuv);
extern void callKernelGL(int width, int height, unsigned char* yuv,  unsigned char* pixels);

extern "C" void bindTexture(cudaArray* array);


#endif