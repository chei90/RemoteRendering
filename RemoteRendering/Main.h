/*************************************************************************

Copyright 2014 Christoph Eichler

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.                                                                    

*************************************************************************/

#pragma once

#define _USE_MATH_DEFINES // for C++
#include <cmath>
#include "RemoteEncoderAPI.h"
#include <Windows.h>
#include <iostream>
#include <stdlib.h>
#include <GL\glew.h>
#include <NVEncoderAPI.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <nvcuvid.h>
#include <iostream>
#include <glm.hpp>
#include <gtc\matrix_inverse.hpp>
#include <gtc\matrix_transform.hpp>
#include <gtc\type_ptr.hpp>
#include <GL\freeglut.h>
#include <sstream>
#include <string>
#include "Geometry.h"
#include "Camera.h"
#include "Util.h"
#include "ConfigFile.h"




#define CUDA_SAFE_CALLING(_error) if(_error != cudaSuccess) {printf("CUDA Error: %s:%i: %s\n", __FILE__, __LINE__, cudaGetErrorString(_error));	}
#define DEFAULT_BUFLEN 512
#define DEFAULT_PORT 8080
#define DEFAULT_IP "127.0.0.1"
#define MOVESPEED 0.05f


using namespace std;

int programID;
Geometry* earth;
int modelLocation;
glm::mat4x4 modelMatrix;
glm::vec3 moveDir;
bool bContinue;
Camera* cam;
bool culling;
bool wireframe;
int n;
int viewProjLoc;
GLuint earthTex;


//Kommunikation zwischen OpenGL und CUDA
GLuint pbo;
cudaGraphicsResource *resource;
cudaDeviceProp prop;
int dev;
int width = 0;
int height = 0;

bool m_continue = true;

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


//Cuda Zeugs
unsigned char* devPtr;
vector<unsigned char> yuv;
unsigned char* d_yuv;
size_t arraySize;

int countFrame;
