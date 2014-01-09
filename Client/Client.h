#ifndef CLIENT_H_
#define CLIENT_H_

#include <string.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/glut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include "Magicnumbers.h"
#include "RenderSocket.h"
#include "Decoder.h"
#include <cudaGL.h>
#include <nvcuvid.h>
#include <helper_cuda_drvapi.h>
#include <vector>
#include "UdpSocket.h"

#include <winsock2.h>
#include <windows.h>

#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>


#include "Decoder.h"
#include "NV12toRGB.h"
// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")


#define CUDA_SAFE_CALLING(_error) if(_error != cudaSuccess) {printf("CUDA Error: %s:%i: %s\n", __FILE__, __LINE__, cudaGetErrorString(_error));	}
#define DEFAULT_BUFLEN 512
#define MAXSTR 512
#define DEFAULT_PORT 8080		// Default Port des Servers
#define DEFAULT_IP "127.0.0.1"	// Default IP Adresse es Servers

int keyStates[256];
int tmpKeyStates[256];
int keySpecialStates[246];
int tmpKeySpecialStates[246];

bool m_continue = true;
bool m_firstFrame = true;
bool m_updateCSC = true;

int m_width = 800, m_height = 600;

CUcontext m_ctx;
CUvideoctxlock m_lock;
FrameQueue* m_queue = 0;
CUdevice m_device;
Decoder* m_decoder;

GLuint currentFrameTex;
cudaGraphicsResource_t cudaTex;
unsigned char* globalMem;

#endif