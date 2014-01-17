#define __COMPILING_DLL
#include "RemoteEncoderAPI.h"
#include "RemoteEncoder.h"
#include "RGBtoYV12.h"
#include "UdpSocket.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_d3d11_interop.h>
#include <vector>
#include "MagicNumbers.h"

RREncoderDesc g_desc;
RemoteEncoder* g_encoder;
CUcontext g_cuCtx;
int g_cuDevice;
cudaGraphicsResource_t g_res;
unsigned char* g_dyuv;
std::vector<unsigned char> g_yuv;
UdpSocket* g_serverSock; 

KeyBoardHandler g_keyHandler;
MouseHandler g_mouseHandler;

bool CM_API RRInit(RREncoderDesc& desc)
{
	//Register UdpSocket
	g_desc = desc;
	//g_encoder = new RemoteEncoder(desc.w, desc.h);
	g_mouseHandler = desc.mouseHandler;
	g_keyHandler = desc.keyHandler;
	//Init Cuda for GL

	cuInit(0);
	g_cuDevice = 0;
	if(g_desc.gfxapi == GL)
	{
		cudaError_t res = cudaGLSetGLDevice(g_cuDevice);
		printf("After cudevice %d\n", res);
	}
	else
	{
		cudaSetDevice(g_cuDevice);
	} 
	//cuCtxCreate(&g_cuCtx, CU_CTX_BLOCKING_SYNC, g_cuDevice);	



	//Allocating Buffers
	g_yuv = std::vector<unsigned char>(g_desc.w * g_desc.h * 3 / 2);
	cudaError_t res = cudaMalloc((void**) &g_dyuv, g_desc.w * g_desc.h * 3 /  2 * sizeof(char));
	printf("ERROR CODE %d\n", res);
	g_serverSock = new UdpSocket();
	g_serverSock->Create();
	g_serverSock->Bind(desc.ip, desc.port);
	//g_encoder->setClientUdp(g_serverSock);

	return true;
}

bool CM_API RRDelete(void)
{
	g_yuv.clear();
	cudaFree((void**) &g_dyuv);
	delete g_encoder;
	return true;
}

void CM_API RRSetSource(void* ptr)
{
	if(g_desc.gfxapi = D3D)
	{
		ID3D11Resource* d11resource = (ID3D11Resource*) ptr;
		cudaGraphicsD3D11RegisterResource(&g_res, d11resource, cudaGraphicsRegisterFlagsNone);
		printf("Mode D3D\n");
	}
	else
	{
		printf("Mode: GL\n");
		GLuint pbo = *((GLuint*) ptr);
		printf("API PBO: %d\n", pbo);
		cudaError_t res = cudaGraphicsGLRegisterBuffer(&g_res, pbo, cudaGraphicsRegisterFlagsReadOnly);
		if(res != cudaSuccess)
		{
			printf("error occured due registering %u\n", res);
		}
	}
}

void CM_API RREncode(void)
{
	unsigned char* devPtr;
	cudaError_t res = cudaGraphicsMapResources(1, &g_res, NULL);
	if(res != cudaSuccess)
	{
		//printf("Something went wrong due mapping");
	}
	res = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, NULL, g_res);
	if(res == cudaSuccess)
	{
		//printf("Something went wrong at creating pointer");
	}
	callKernel(g_desc.w,g_desc.h,g_dyuv, devPtr);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &g_res, NULL);
	cudaMemcpy( &g_yuv[0], g_dyuv,  g_yuv.size(), cudaMemcpyDeviceToHost);
	g_encoder->setPicBuf(&g_yuv[0]);
	g_encoder->encodePB();
}

const RREncoderDesc& RRGetDesc(void)
{
	return g_desc;
}

void CM_API RRWaitForConnection()
{
	char message[64];
	g_serverSock->Receive(message, 64);
	UINT8 identifier;
	memcpy(&identifier, message, sizeof(UINT8));

	std::cout << "Client connected" << std::endl;

	if(identifier == WINDOW_SIZE) 
	{
		// later
	}

	g_serverSock->SetToNonBlock();
}

void CM_API RRQueryClientEvents()
{
	static char msg[64];
	memset(msg, 0, 64);
	g_serverSock->Receive(msg, 64);

	int key = 0, identifyer = 0;
	memcpy(&identifyer, msg, sizeof(UINT8));
	switch(identifyer)
	{
	case KEY_PRESSED:
		memcpy(&key, msg+sizeof(UINT8), sizeof(int));
		cout << "KEY HIT: " << key << endl;
		if(key <= 256)
			g_keyHandler(key, true);
		break;
	case KEY_RELEASED:
		memcpy(&key, msg+sizeof(UINT8), sizeof(int));
		cout << "KEY RELASED!" << key << endl;
		if(key <= 256)
			g_keyHandler(key, false);
		break;
	case SPECIAL_KEY_PRESSED:
		memcpy(&key, msg+sizeof(UINT8), sizeof(int));
		cout << "SPECIAL KEY Pressed!" << key << endl;
		if(key <= 246)
			//keySpecialStates[key] = true;
				break;
	case SPECIAL_KEY_RELEASED:
		memcpy(&key, msg+sizeof(UINT8), sizeof(int));
		cout << "SPECIAL KEY RELASED!" << key << endl;
		if(key <= 246)
			//keySpecialStates[key] = false;
				break;
	default:
		break;
	}
}