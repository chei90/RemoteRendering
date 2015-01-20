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

#include <GL/glew.h>
#include <GL/freeglut.h>

#ifndef PRIMITIVE_RESTART
#define PRIMITIVE_RESTART -1
#endif

class Geometry
{
public:
	Geometry(void);
	~Geometry(void);
	
	static const int ATTR_POS = 0;
	static const int ATTR_NORMAL = 1;
	static const int ATTR_COLOR = 2;
	static const int ATTR_TEX_COORDS = 3;

	void setIndexBuffer(int* indices, int topology, int count);
	void setVertices(float* vertices, int count);
	void construct();
	void draw();

private:
	GLuint vaid;
	float* vertexValueBuffer;
	int* indexValueBuffer;
	int topology;
	int indexCount;
	int vertexCount;
	GLuint vbid;
	GLuint ibid;
	bool constructed;
};

