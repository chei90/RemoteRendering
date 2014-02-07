#ifndef RGBTOYV12_H_
#define RGBTOYV12_H_
#include <Windows.h>
#include <cuda.h>
#include <cuda_runtime.h>

//Calls the Kernel for D3D Conversion
extern void callKernelD3D(int width, int height, unsigned char* yuv);
//Calls the Kernel for GL Conversion
extern void callKernelGL(int width, int height, unsigned char* yuv,  unsigned char* pixels);

//Binds D3D Texture to Array
extern "C" void bindTexture(cudaArray* array);

#endif