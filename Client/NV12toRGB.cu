#ifndef RGBTOYV12_CU_
#define RGBTOYV12_CU_
#include "NV12toRGB.h"


__device__ int clamp (int arg, int minVal, int maxVal)
{
	return max(minVal, min(arg, maxVal));
}

__global__ void NV12toRGB(unsigned char* nv12, unsigned char* rgba, int decodedPitch)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int i = iy * decodedPitch + ix;
	int rgbStart = (iy * gridDim.x * blockDim.x + ix) * 4;

	int quadX = (ix / 2);
	int quadY = (iy / 2);

	int uvAdr = decodedPitch / 2 * quadY + quadX;
	int uvStart = decodedPitch * gridDim.y * blockDim.y;

	int y = nv12[i];
	int u = nv12[uvStart + 2 * uvAdr];
	int v = nv12[uvStart + 2 * uvAdr + 1];

	int c = y - 16;
	int d = u - 128;
	int e = v - 128;

	// R
	int r = clamp(( 298 * c           + 409 * e + 128) >> 8, 0, 255);
	//int r = 1.164 * (y-16) + 1.1596 * (v-128);
	// G
	int g = clamp(( 298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255);
	//int g = 1.164 * (y-16) - 0.813 * (v - 128) - 0.391 * (u - 128);
	// B
	int b = clamp(( 298 * c + 516 * d           + 128) >> 8, 0, 255);
	//int b = 1.164 * (y-16) + 2.018 * (u - 128);


	rgba[rgbStart] = r;
	rgba[rgbStart+1] = g;
	rgba[rgbStart+2] = b;
	rgba[rgbStart+3] = 255;
	
}


void callDecode(int width, int height, unsigned char* nv12, unsigned char* globalMem, int decodedPitch)
{
	dim3 blocks (8, 8);
	dim3 numBlocks (width / 8, height / 8);
	//gpuErrchk(cudaGetLastError());
	NV12toRGB<<<numBlocks, blocks>>>(nv12, globalMem, decodedPitch);
	//gpuErrchk(cudaGetLastError());
	cudaDeviceSynchronize();

}

#endif