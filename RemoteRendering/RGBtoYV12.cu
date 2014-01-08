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

	unsigned char y = 0.299 * r + 0.587 * g + 0.114 * b;
	yuv[upos - (row+1)*iwidth + col] = y;


     if (           !((i/gridDim.x)%2)             &&        !(i%2))
     {

        //YV12
 	    // U
		yuv[width - ( (iwidth/2) * ((row/2)+1) - ((col/2)+1)  )] = 0.493 * (b - y);//((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
        // V
		yuv[vpos - ( (iwidth/2) * ((row/2)+1) - ((col/2)+1)  )] = 0.887 * (r - y); //((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;

    }
}


extern void callKernel(int width, int height, unsigned char* yuv, unsigned char* devPtr)
{
	RGBtoYV12<<<width, height>>>(yuv, devPtr);
    cudaDeviceSynchronize();
}

#endif

