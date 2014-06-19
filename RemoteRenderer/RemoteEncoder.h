#ifndef REMOTEENCODER_H_
#define REMOTEENCODER_H_

#include "UdpSocket.h"
#include <Windows.h>
#include <NVEncoderAPI.h>
#include <cuda.h>
#include <NVEncodeDataTypes.h>
#include <nvcuvid.h>
#include <iostream>
#include "types.h"



#define ASPECT_WIDTH 4
#define ASPECT_HEIGHT 3

using namespace std;

class RemoteEncoder
{

public:
	RemoteEncoder(int o_width, int o_height);
	~RemoteEncoder();

	//Encoding
	bool encodePB();
	void handleCudaError(CUresult cuRes, const char* c);
	void setDevicePtr(CUdeviceptr dptr);

	//Getter / Setter
	unsigned char *getCharBuf(){return outBuf;}
	void setClientUdp(UdpSocket* c){client = c;}
	UdpSocket* getClient(){return client;}
	void incPicID(){picId = (picId++) % 256;}
	UINT8 getPicId(){return picId;}
	int getWidth() {return width;}
	int getHeight() {return height;}
	void setPicBuf(unsigned char* buf){m_efParams.picBuf = buf;}

private:

	//Error Handling
	void handleHR(HRESULT hr, const char* c);
	//Create Cuda
	void createCuda();
	//Set Params
	void setEncoderParams();
	//Set Callbacks
	bool setCBFunctions(NVVE_CallbackParams *pCB, void *pUserData);



	// Encoding Stuff
	NVEncoderParams*	m_EncoderParams;
	NVVE_CallbackParams m_cbParams;
	NVVE_EncodeFrameParams m_efParams;
	NVVE_CallbackParams m_NVCB;

	//Cuda
	CUdevice m_cuDevice;
	CUcontext m_cuContext;
	CUvideoctxlock m_cuCtxLock;
	CUdeviceptr dptr;
	void* m_CudaEncoder;

	//General Stuff
	int width, height;
	UINT8 picId;
	UdpSocket* client;
	unsigned char* outBuf;
	HRESULT errorHandling;
	bool latencyMeasure;
};

#endif
