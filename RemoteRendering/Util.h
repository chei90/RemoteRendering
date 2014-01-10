/*
 * Util.hpp
 *
 *  Created on: 31.07.2013
 *      Author: christoph
 */
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
Geometry* createSphere(float r, int n, int k, const char* imageFile);
float*** getImage(const char* fileName, int* height, int* width);
void glCheckError(GLenum error, const char* msg);

