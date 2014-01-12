#ifndef RGBTOYV12_CU_
#define RGBTOYV12_CU_

#include "RGBtoYV12.h"


// Dieser Kernel wird mit
// RGBtoYV12<<<gridSize, blockSize>>>(yuv, devPtr);
//aufgerufen.

//er berechnet direkt die YUV-Werte aus dem RGBA- Format
// (Hoffentlich nun direkt richtigrum und nicht mehr spiegelverkehrt.

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

	float y = (0.257 * r) + (0.504 * b) + (0.098 * b) + 16;//0.299 * r + 0.587 * g + 0.114 * b;
	yuv[upos - (row+1)*iwidth + col] = y;


     if (           !((i/gridDim.x)%2)             &&        !(i%2))
     {

        //YV12
 	    // U
		float u = -1 * (0.148 * r) - (0.291 * g) + (0.439 * b) + 128;//0.493 * (b - y);
		yuv[width - ( (iwidth/2) * ((row/2)+1) - ((col/2)+1)  )] = u;//((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        // V
		float v = (0.439 * r) - (0.368 * g) - (0.071 * b) + 128;//0.887 * (r - y);
		yuv[vpos - ( (iwidth/2) * ((row/2)+1) - ((col/2)+1)  )] = v; //((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;

    }
}


extern void callKernel(int width, int height, unsigned char* yuv, unsigned char* devPtr)
{
	RGBtoYV12<<<width, height>>>(yuv, devPtr);
    cudaDeviceSynchronize();
}

#endif

