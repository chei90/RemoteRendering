#ifndef MAIN_H_
#define MAIN_H_
#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include "RenderSocket.h"
#include <Windows.h>
#include <iostream>
#include <stdlib.h>
#include <gl\glew.h>
#include <gl\freeglut.h>
#include <gl\GL.h>
#include <gl\GLU.h>
#include <NVEncoderAPI.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvcuvid.h>
#include <iostream>
#include <GL\glew.h>
#include "glm.hpp"
#include <gtc\matrix_inverse.hpp>
#include <gtc\matrix_transform.hpp>
#include <gtc\type_ptr.hpp>
#include <GL\freeglut.h>
#include <GL\gl.h>
#include <GL\glu.h>
#include <sstream>

#include <boost\shared_ptr.hpp>
#include <boost\regex.hpp>
#include <boost\foreach.hpp>
#include <boost\property_tree\ptree.hpp>
#include <boost\property_tree\ptree_fwd.hpp>
#include <boost\property_tree\json_parser.hpp>
#include <boost\property_tree\json_parser.hpp>


#include "Modell.h"
#include "Util.h"

#include "RemoteEncoder.h"
#include "RGBtoYV12.h"



#define CUDA_SAFE_CALLING(_error) if(_error != cudaSuccess) {printf("CUDA Error: %s:%i: %s\n", __FILE__, __LINE__, cudaGetErrorString(_error));	}
#define DEFAULT_BUFLEN 512
#define DEFAULT_PORT 8080
#define DEFAULT_IP "127.0.0.1"


using namespace std;
using boost::property_tree::ptree;

GLuint vaoId = 0;
GLuint vboId = 0;
GLuint iboId = 0;
GLuint naoId = 0;
GLuint nboId = 0;
GLuint framebuffer;
GLuint programID = 0;
GLuint ATTR_POS;
GLuint ATTR_NORMAL;

//Kommunikation zwischen OpenGL und CUDA
GLuint pbo;
cudaGraphicsResource *resource;
cudaDeviceProp prop;
int dev;
RemoteEncoder* remo;

vector<int> vbo;
vector<int> ibo;
vector<int> nbo;
vector<Modell*> part;

int width = 0;
int height = 0;
float factor = 0.0004f;

//Shader Stuff
float translation_X = 0.0f;
float translation_Y = 0.0f;
float zoom = -4.0f;
float rotationAngle_X = 18.0f;
float rotationAngle_Y = 0.0f;

bool vw = false;
bool m_continue = true;

int partID;

//User Input
bool keyStates[256];
bool keySpecialStates[246];
bool lastState;

int recentMouse_X;
int recentMouse_Y;
int mouseButton = 0;
int mousePressed = 1;

CUcontext cuCtx;
CUdevice cuDev;

SYSTEMTIME st;
DWORD currentTimeMS;
DWORD lastTimeMS;

FILE* f;

RenderSocket* serverSocket;
RenderSocket* client;

//Cuda Zeugs
unsigned char* devPtr;
vector<unsigned char> yuv;
unsigned char* d_yuv;
size_t arraySize;

int countFrame;

#endif
