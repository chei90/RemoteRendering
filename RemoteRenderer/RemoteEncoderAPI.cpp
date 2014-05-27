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
#include "TcpSocket.h"

RREncoderDesc g_desc;
RemoteEncoder* g_encoder;
CUcontext g_cuCtx;
int g_cuDevice;
cudaGraphicsResource_t g_res;
unsigned char* g_dyuv;
std::vector<unsigned char> g_yuv;
TcpSocket* g_serverSock; 
TcpSocket* clientSock;
bool g_keyStates[256];
int width, height;

int latencyCnt;
bool latencyMeasure;

typedef void (*encodeCB)(void);

encodeCB g_encodeCB;
KeyBoardHandler g_keyHandler;
MouseHandler g_mouseHandler;

void encodeD3D()
{
	cudaArray_t devptr = NULL;

	cudaGraphicsSubResourceGetMappedArray((cudaArray_t*)&devptr, g_res, 0, 0);
	bindTexture((cudaArray_t)devptr);
	callKernelD3D(width, height, g_dyuv);
}

void encodeGL()
{
	unsigned char* devPtr = NULL;
	cudaGraphicsResourceGetMappedPointer((void**)&devPtr, NULL, g_res);
	callKernelGL(width, height, g_dyuv, devPtr);
}

bool CM_API RRInit(RREncoderDesc& desc)
{
	//Register UdpSocket
	g_desc = desc;
	width = g_desc.w;
	height = g_desc.h;

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
	g_serverSock = new TcpSocket();
	clientSock = new TcpSocket();
	g_serverSock->Create();
	g_serverSock->Bind(g_desc.ip, g_desc.port);
	g_encoder->setClientTcp(clientSock);

	latencyMeasure = false;
	latencyCnt = 0;
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
	unsigned char* devPtr = NULL;
	cudaGraphicsMapResources(1, &g_res, NULL);
	g_encodeCB();
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &g_res, NULL);
	cudaMemcpy(&g_yuv[0], g_dyuv, g_yuv.size(), cudaMemcpyDeviceToHost);
	if(latencyMeasure && latencyCnt++ == 1)
	{
		memset(&g_yuv[0], 0, g_yuv.size());
		latencyCnt=0;
		latencyMeasure=false;
	}
	g_encoder->setPicBuf(&g_yuv[0]);
	g_encoder->setMeasure(latencyMeasure);
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
	bool list = g_serverSock->Listen(1);

	if(g_serverSock->Accept(*clientSock))
		printf("Verbindung Fehlgeschlagen!\n");
	printf("Client erfolgreich verbunden!\n");
	int i = clientSock->Receive(message, 64);
	printf("%d signs received \n", i);
	UINT8 identifier = 0;
	memcpy(&identifier, message, sizeof(UINT8));
	printf("Identifyer is %d \n", identifier);
	clientSock->SetToNonBlock();
}

void CM_API RRQueryClientEvents()
{
	static char msg[64];
	memset(msg, 0, 64);
	clientSock->Receive(msg, 64);

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
			latencyMeasure = true;
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