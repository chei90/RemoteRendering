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

#include <Windows.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "Geometry.h"
#include <stdio.h>
#include <tchar.h>
#include <gtc\matrix_inverse.hpp>
#include <gtc\matrix_transform.hpp>
#include <gtc\type_ptr.hpp>
#include <gtx/rotate_vector.hpp>
#include <IL/il.h>
#include <vector>

#define _USE_MATH_DEFINES
#include <math.h>

std::string textFileRead(const char* filePath);
int createShaderProgram(const char* vs, const char* fs);
Geometry* createSphere(float r, int n, int k);
GLuint createTexture(const wchar_t* fileName);
float* getImage(const wchar_t* fileName, int* height, int* width, int* imgFormat);
void glCheckError(GLenum error, const char* msg);

