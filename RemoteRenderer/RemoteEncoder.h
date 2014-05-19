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
	bool SetCBFunctions(NVVE_CallbackParams *pCB, void *pUserData);
	//Initialize & Errorhandling


	//Encoding
	bool encode();
	bool encodePB();
	void handleCudaError(CUresult, const char* c);
	void setDevicePtr(CUdeviceptr dptr);

	unsigned char *GetCharBuf()
	{
		return outBuf;
	}

	FILE *GetFileOut()
	{
		return out;
	}
	void setClientUdp(UdpSocket* c)
	{
		client = c;
	}
	UdpSocket* getClient()
	{
		return client;
	}

	void setMeasure(bool measure)
	{
		this->latencyMeasure = measure;
	}

	bool getMeasure()
	{
		return this->latencyMeasure;
	}

	void incPicID()
	{
		picId = (picId++) % 256;
	}

	UINT8 getPicId()
	{
		return picId;
	}
	int getWidth() {return width;}
	int getHeight() {return height;}
	void setPicBuf(unsigned char* buf){m_efParams.picBuf = buf;}

private:

	void handleHR(HRESULT hr, const char* c);
	void createCuda();
	void setEncoderParams();
	CUdevice getCudaDevice();
	// Encoding Stuff
	NVEncoderParams*	m_EncoderParams;
	int width, height;
	//out
	UINT8 picId;

	UdpSocket* client;
	FILE* out;
	//unsigned char* encodedFrame;


	//NVEncoder m_CudaEncoder;
	void* m_CudaEncoder;
	NVVE_CallbackParams m_cbParams;
	NVVE_EncodeFrameParams m_efParams;
	CUdevice m_cuDevice;
	CUcontext m_cuContext;
	CUvideoctxlock m_cuCtxLock;
	CUdeviceptr dptr;
	// Buffer
	unsigned char* m_VideoFrame;
	unsigned char* outBuf;

	//Callbacks
	NVVE_CallbackParams m_NVCB;
	//void *m_pEncoder;
	// Errorhandling
	HRESULT errorHandling;
	bool latencyMeasure;
};

#endif
