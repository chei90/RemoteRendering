#ifndef RGBTOYV12_CU_
#define RGBTOYV12_CU_

#include "RGBtoYV12.h"

//Clamps float to range of minv, maxv
__device__ float clamp(float v, float minv, float maxv)
{
    return max(min(maxv, v), minv);
}

//Clamps float4 to range of minv, maxv
__device__ float4 clamp(float4 v, float minv, float maxv)
{
    return make_float4(clamp(v.x, minv, maxv), clamp(v.y, minv, maxv), clamp(v.z, minv, maxv), clamp(v.w, minv, maxv));
}

//Converts D3D Texture to YV12
texture<float4, cudaTextureType2D, cudaReadModeElementType> g_d3dSurface;
__global__ void RGBtoYV12D3D(unsigned char* yuv)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int gridWidth = blockDim.x * gridDim.x;
	int i = iy * gridWidth + ix;
	int globalBufferSize = gridDim.x * blockDim.x * gridDim.y * blockDim.y * 1.5;
 	int rgbID = i * 4;
    int upos = blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int vpos = upos + upos / 4;

    float4 color = tex2D(g_d3dSurface, ix, iy);
    color = clamp(color, 0, 1);
 	int r = (int)(255 * color.x), g = (int)(255 * color.y), b = (int)(255 * color.z);


	int y =  ((  66 * r + 129 * g +  25 * b + 128) >> 8) +  16;
	yuv[upos - (iy+1)*gridWidth + ix] = y;

    if (!((i/gridWidth)%2) && !(i%2))
    {
 	    // U
		int u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
		yuv[globalBufferSize - ((gridWidth/2) * ((iy/2)+1) - ((ix/2)+1))] = u;
        // V
		int v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
		yuv[vpos - ((gridWidth/2) * ((iy/2)+1) - ((ix/2)+1))] = v; 
    }
}

//Converts GL Buffer to YV12
__global__ void RGBtoYV12GL(unsigned char* yuv, unsigned char* pData)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int gridWidth = blockDim.x * gridDim.x;
	int i = iy * gridWidth + ix;
	int globalBufferSize = gridDim.x * blockDim.x * gridDim.y * blockDim.y * 1.5;
 	int rgbID = i * 4;
    int upos = blockDim.x * gridDim.x * blockDim.y * gridDim.y;
    int vpos = upos + upos / 4;
 	int r = pData[rgbID], g = pData[rgbID+1], b = pData[rgbID+2];


	int y =  ((  66 * r + 129 * g +  25 * b + 128) >> 8) +  16;
	yuv[upos - (iy+1)*gridWidth + ix] = y;

    if (!((i/gridWidth)%2) && !(i%2))
    {
 	    // U
		int u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
		yuv[globalBufferSize - ((gridWidth/2) * ((iy/2)+1) - ((ix/2)+1))] = u;
        // V
		int v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
		yuv[vpos - ((gridWidth/2) * ((iy/2)+1) - ((ix/2)+1))] = v; 
    }
}

extern void callKernelGL(int width, int height, unsigned char* yuv,  unsigned char* devPtr)
{
	dim3 numBlocks(width / 8, height / 8);
	dim3 numThreads(8 , 8);

	RGBtoYV12GL<<<numBlocks, numThreads>>>(yuv, devPtr);
    cudaDeviceSynchronize();
}

extern void callKernelD3D(int width, int height, unsigned char* yuv)
{
	dim3 numBlocks(width / 8, height / 8);
	dim3 numThreads(8 , 8);

	RGBtoYV12D3D<<<numBlocks, numThreads>>>(yuv);
    cudaDeviceSynchronize();
}

extern void bindTexture(cudaArray* array)
{
    g_d3dSurface.addressMode[0] = cudaAddressModeMirror;
    g_d3dSurface.addressMode[1] = cudaAddressModeMirror;
    g_d3dSurface.addressMode[2] = cudaAddressModeMirror;
    g_d3dSurface.normalized = 0;
    g_d3dSurface.filterMode = cudaFilterModeLinear;
    cudaBindTextureToArray(g_d3dSurface, array);
}

#endif

