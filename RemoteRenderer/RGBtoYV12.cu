#ifndef RGBTOYV12_CU_
#define RGBTOYV12_CU_

#include "RGBtoYV12.h"


__device__ float clamp(float v, float minv, float maxv)
{
    return max(min(maxv, v), minv);
}

__device__ float4 clamp(float4 v, float minv, float maxv)
{
    return make_float4(clamp(v.x, minv, maxv), clamp(v.y, minv, maxv), clamp(v.z, minv, maxv), clamp(v.w, minv, maxv));
}


// Dieser Kernel wird mit
// RGBtoYV12<<<gridSize, blockSize>>>(yuv, devPtr);
//aufgerufen.

//er berechnet direkt die YUV-Werte aus dem RGBA- Format
// (Hoffentlich nun direkt richtigrum und nicht mehr spiegelverkehrt.

texture<float4, cudaTextureType2D, cudaReadModeElementType> g_d3dSurface;

__global__ void RGBtoYV12D3D(unsigned char* yuv)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int width = gridDim.x * blockDim.x * 1.5;
	//bildbreite
	int iwidth = gridDim.x;

 	int rgbID = i * 4;
    int upos = blockDim.x * gridDim.x;
    int vpos = upos + upos / 4;
	int col = i % iwidth;
	int row = i / iwidth; //bzw. threadIdx.x;

    float4 color = tex2D(g_d3dSurface, col, row);
    color = clamp(color, 0, 1);

 	int r = (int)(255 * color.x), g = (int)(255 * color.y), b = (int)(255 * color.z);


	//Y
	// neu (hoffentlich richtig rum und nicht gespiegelt)
	// nur noch zeilenweise umgedreht. die eigene ID steckt in
	// den Zeilen und Spalten, und taucht deshalb nicht extra auf.

	int y =  ((  66 * r + 129 * g +  25 * b + 128) >> 8) +  16;
	yuv[upos - (row+1)*iwidth + col] = y;


     if (           !((i/gridDim.x)%2)             &&        !(i%2))
     {

        //YV12
 	    // U
		int u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
		yuv[width - ( (iwidth/2) * ((row/2)+1) - ((col/2)+1)  )] = u;
        // V
		int v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
		yuv[vpos - ( (iwidth/2) * ((row/2)+1) - ((col/2)+1)  )] = v; 

    }
}


__global__ void RGBtoYV12(unsigned char* yuv, unsigned char* pData)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
    int width = gridDim.x * blockDim.x * 1.5;
	//bildbreite
	int iwidth = gridDim.x;

 	int rgbID = i * 4;
    int upos = blockDim.x * gridDim.x;
    int vpos = upos + upos / 4;
	int col = i % iwidth;
	int row = i / iwidth; //bzw. threadIdx.x;

 	int r = pData[rgbID], g = pData[rgbID+1], b = pData[rgbID+2];


	//Y
	// neu (hoffentlich richtig rum und nicht gespiegelt)
	// nur noch zeilenweise umgedreht. die eigene ID steckt in
	// den Zeilen und Spalten, und taucht deshalb nicht extra auf.

	int y =  ((  66 * r + 129 * g +  25 * b + 128) >> 8) +  16;
	yuv[upos - (row+1)*iwidth + col] = y;


     if (           !((i/gridDim.x)%2)             &&        !(i%2))
     {

        //YV12
 	    // U
		int u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
		yuv[width - ( (iwidth/2) * ((row/2)+1) - ((col/2)+1)  )] = u;
        // V
		int v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
		yuv[vpos - ( (iwidth/2) * ((row/2)+1) - ((col/2)+1)  )] = v; 

    }
}

/*extern void callKernel(int width, int height, unsigned char* yuv, unsigned char* devPtr)
{
	RGBtoYV12<<<width, height>>>(yuv, devPtr);
    cudaDeviceSynchronize();
}*/

extern void callKernel(int width, int height, unsigned char* yuv, unsigned char* devPtr)
{
	RGBtoYV12D3D<<<width, height>>>(yuv);
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

