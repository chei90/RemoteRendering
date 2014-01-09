#include "Geometry.h"
#include <memory>

Geometry::Geometry(void)
{
	constructed = false;
}

Geometry::~Geometry(void)
{
	glDeleteBuffers(1, &vaid);
	glDeleteBuffers(1, &ibid);
	glDeleteBuffers(1, &vbid);

	vaid = -1;
	ibid = vbid = 0;
	constructed = false;
}

void Geometry::setIndexBuffer(int* indices, int topology, int count)
{
	indexValueBuffer = new int[count];
	memcpy(indexValueBuffer, indices, count * sizeof(int));
	this->topology = topology;
	indexCount = count;
}

void Geometry::setVertices(float* vertices, int count)
{
	vertexValueBuffer = new float[count];
	memcpy(vertexValueBuffer, vertices, sizeof(float) * count);
	vertexCount = count;
}

void Geometry::construct()
{
	GLenum error = glGetError();
	constructed = true;
	glGenVertexArrays(1, &vaid);
	error = glGetError();
	glGenBuffers(1, &vbid);
	error = glGetError();
	glGenBuffers(1, &ibid);
	error = glGetError();

	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(-1);
	glBindVertexArray(vaid);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibid);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indexCount, indexValueBuffer, GL_STATIC_DRAW);
	error = glGetError();
	glBindBuffer(GL_ARRAY_BUFFER, vbid);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertexCount, vertexValueBuffer, GL_STATIC_DRAW);
	error = glGetError();
	glEnableVertexAttribArray(ATTR_POS);
	glEnableVertexAttribArray(ATTR_COLOR);
	error = glGetError();
	glVertexAttribPointer(ATTR_POS, 3, GL_FLOAT, false, 6*sizeof(GLfloat), 0);
	glVertexAttribPointer(ATTR_COLOR, 3, GL_FLOAT, false, 6*sizeof(GLfloat), (GLvoid*) (3*sizeof(GLfloat)));
	error = glGetError();
	glBindVertexArray(0);
}

void Geometry::draw()
{
	if(!constructed)
	{
		construct();
	}
	glBindVertexArray(vaid);
	glDrawElements(topology, indexCount, GL_UNSIGNED_INT, 0);
}