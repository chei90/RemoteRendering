#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>


class Geometry
{
public:
	Geometry(void);
	~Geometry(void);

	static const int ATTR_POS = 0;
	static const int ATTR_NORMAL = 1;
	static const int ATTR_COLOR = 2;

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

