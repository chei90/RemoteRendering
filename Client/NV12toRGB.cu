#ifndef RGBTOYV12_CU_
#define RGBTOYV12_CU_
#include "NV12toRGB.h"


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

	// R
	float r = y + 1.13983 * v;
	//int r = 1.164 * (y-16) + 1.1596 * (v-128);
	// G
	float g = y - 0.39393 * u - 0.58081 * v;
	//int g = 1.164 * (y-16) - 0.813 * (v - 128) - 0.391 * (u - 128);
	// B
	float b = y + 2.028 * u;
	//int b = 1.164 * (y-16) + 2.018 * (u - 128);


	rgba[rgbStart] = r;
	rgba[rgbStart+1] = g;
	rgba[rgbStart+2] = b;
	rgba[rgbStart+3] = 255;
	
}


void callDecode(int width, int height, unsigned char* nv12, unsigned char* globalMem, int decodedPitch)
{
	dim3 blocks (10, 10);
	dim3 numBlocks (width / 10, height / 10);
	//gpuErrchk(cudaGetLastError());
	NV12toRGB<<<numBlocks, blocks>>>(nv12, globalMem, decodedPitch);
	//gpuErrchk(cudaGetLastError());
	cudaDeviceSynchronize();

}

#endif