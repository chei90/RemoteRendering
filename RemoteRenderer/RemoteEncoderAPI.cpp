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
bool g_keyStates[256];

KeyBoardHandler g_keyHandler;
MouseHandler g_mouseHandler;


typedef void (*encodeCB)(void);

encodeCB g_encodeCB;

void encodeD3D(void)
{
	cudaArray_t devptr = NULL;
	cudaGraphicsSubResourceGetMappedArray((cudaArray_t*)&devptr, g_res, 0, 0);
	bindTexture((cudaArray_t)devptr);
	callKernelD3D(800, 600, g_dyuv);  
}

void encodeGL(void)
{
	unsigned char* devPtr = NULL;
	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, NULL, g_res);
    callKernelGL(800, 600, g_dyuv, devPtr);
}

bool CM_API RRInit(RREncoderDesc& desc)
{
	//Register UdpSocket
	g_desc = desc;

	g_encoder = new RemoteEncoder(desc.w, desc.h);
	g_mouseHandler = desc.mouseHandler;
	g_keyHandler = desc.keyHandler;
	//Init Cuda for GL

	cuInit(0);
	g_cuDevice = 0;
	if(g_desc.gfxapi == GL)
	{
		cudaError_t res = cudaGLSetGLDevice(g_cuDevice);
        g_encodeCB = encodeGL;
    }
	else
	{
		cudaSetDevice(g_cuDevice);
        g_encodeCB = encodeD3D;
	} 
	cuCtxCreate(&g_cuCtx, CU_CTX_BLOCKING_SYNC, g_cuDevice);	

	//Allocating Buffers
	cuCtxPushCurrent(g_cuCtx);
	g_yuv = std::vector<unsigned char>(g_desc.w * g_desc.h * 3 / 2);
	cudaError_t res = cudaMalloc((void**) &g_dyuv, g_desc.w * g_desc.h * 3 /  2 * sizeof(char));
	cuCtxPopCurrent(NULL);
	g_serverSock = new UdpSocket();
	g_serverSock->Create();
	g_serverSock->Bind(g_desc.ip, g_desc.port);
	g_encoder->setClientUdp(g_serverSock);

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
	cuCtxPushCurrent(g_cuCtx);
	if(g_desc.gfxapi == D3D)
	{
		ID3D11Resource* d11resource = (ID3D11Resource*) ptr;
		cudaError_t e = cudaGraphicsD3D11RegisterResource(&g_res, d11resource, cudaGraphicsRegisterFlagsNone);

        char msg[2048];
        memset(msg, 0, 2048);

        sprintf_s(msg, "%d\n", e);
        OutputDebugStringA(msg);
	}
	else
	{
		GLuint pbo = *((GLuint*) ptr);
		cudaError_t res = cudaGraphicsGLRegisterBuffer(&g_res, pbo, cudaGraphicsRegisterFlagsReadOnly);
		if(res != cudaSuccess)
		{
			printf("error occured due registering %u\n", res);
		}
	}
	cuCtxPopCurrent(NULL);
}

void CM_API RREncode(void)
{
    cuCtxPushCurrent(g_cuCtx);
    //unsigned char* devPtr = NULL;
    cudaError_t res = cudaGraphicsMapResources(1, &g_res, NULL);

    //res = cudaGraphicsResourceGetMappedPointer((void**)&devPtr, NULL, g_res);
    /*cudaArray_t devptr = NULL;

    cudaGraphicsSubResourceGetMappedArray((cudaArray_t*)&devptr, g_res, 0, 0);

    bindTexture((cudaArray_t)devptr);

    callKernelD3D(800, 600, g_dyuv, devPtr);*/

	g_encodeCB();

    res = cudaDeviceSynchronize();

    res = cudaGraphicsUnmapResources(1, &g_res, NULL);

    res = cudaMemcpy(&g_yuv[0], g_dyuv, g_yuv.size(), cudaMemcpyDeviceToHost);

    g_encoder->setPicBuf(&g_yuv[0]);

    g_encoder->encodePB();

    cuCtxPopCurrent(NULL);
}

const RREncoderDesc& RRGetDesc(void)
{
	return g_desc;
}

void CM_API RRWaitForConnection()
{
	std::cout << "Waiting for connection" << std::endl;
		 
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

	int key = 0, identifyer = 0, dx = 0, dy = 0;
	memcpy(&identifyer, msg, sizeof(UINT8));
	switch(identifyer)
	{
	case KEY_PRESSED:
		memcpy(&key, msg+sizeof(UINT8), sizeof(int));
		if(key <= 256)
		{
			g_keyHandler(key, true);
			g_keyStates[key] = true;	
		}
		break;
	case KEY_RELEASED:
		memcpy(&key, msg+sizeof(UINT8), sizeof(int));
		if(key <= 256)
		{
			g_keyHandler(key, false);
			g_keyStates[key] = false;
		}
		break;
	case SPECIAL_KEY_PRESSED:
		memcpy(&key, msg+sizeof(UINT8), sizeof(int));
		if(key <= 246)
			//keySpecialStates[key] = true;
				break;
	case SPECIAL_KEY_RELEASED:
		memcpy(&key, msg+sizeof(UINT8), sizeof(int));
		if(key <= 246)
			//keySpecialStates[key] = false;
				break;
	case MOUSE_PRESSED:
		memcpy(&dx, msg + sizeof(UINT8), sizeof(int));
		memcpy(&dy, msg + sizeof(UINT8) + sizeof(int), sizeof(int));
		g_mouseHandler(dx, dy, 0, 1);
		break;
	case MOUSE_RELEASED:
		g_mouseHandler(0, 0, 0, 0);
		break;
	default:
		break;
	}
}

bool CM_API RRIsKeyDown(char key)
{
	return g_keyStates[key];
}