#ifndef NV12TORGB_H_
#define NV12TORGB_H_

#include <Windows.h>
#include <cuda.h>
#include <cuda_surface_types.h>
#include <stdio.h>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
	  fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
	  std::cin.get();
	  if (abort) exit(code);
   }
}

extern void callDecode(int width, int height, unsigned char* nv12, unsigned char* globalMem, int decodedPitch);


#endif