#pragma once

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm.hpp>
#include <gtc\matrix_inverse.hpp>
#include <gtc\matrix_transform.hpp>
#include <gtc\type_ptr.hpp>
#include "Geometry.h"
#include "Camera.h"
#include "Util.h"


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


void render()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glUseProgram(programID);

	glUniformMatrix4fv(modelLocation, 1, false, glm::value_ptr(modelMatrix));
	earth->draw();
	glm::mat4 viewProj = cam->getProjection() * cam->getView();
	glUniformMatrix4fv(viewProjLoc, 16, false, glm::value_ptr(viewProj));

	glutPostRedisplay();
	glutSwapBuffers();
}

void initialize(int argc, char** argv)
{
	modelMatrix = glm::mat4x4();
	moveDir = glm::vec3(0.0f, 0.0f, 0.0f);
	bContinue = true;
	cam = new Camera();
	culling = true;
	wireframe = true;
	n = 2;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(800, 600);
	glutCreateWindow("Decoder!");
	glewInit();	

	glutDisplayFunc(render);
	
	glViewport(0, 0, 800, 600);
	glPrimitiveRestartIndex(-1);
	glEnable(GL_PRIMITIVE_RESTART);
}

void main(int argc, char** argv)
{
	initialize(argc, argv);
	glCheckError(glGetError(), "After init");
	programID = createShaderProgram("shader/Main_VS.glsl", "shader/VertexColor_FS.glsl");
	glCheckError(glGetError(), "After creating Program");
	modelLocation = glGetUniformLocation(programID, "model");
	glCheckError(glGetError(), "After model");
	viewProjLoc = glGetUniformLocation(programID, "viewProj");
	glCheckError(glGetError(), "After ViewProjLoc");
	cam->move(0.0f, 0.0f, 5.0f);
	earth = createSphere(2, n, 2*n, "textures/earth.jpg");

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glClearColor(0.1f, 0.0f, 0.0f, 1.0f);
	glCheckError(glGetError(), "Before main Loop");
	glutMainLoop();
	std::cin.get();
}
