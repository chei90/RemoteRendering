#include "Camera.h"
#include <stdlib.h>
#include <iostream>

Camera::Camera(void)
{
	phi = 0;
	theta = 0;
	viewDir = glm::vec3(0,0,1);
	upDir = glm::vec3(0,1,0);
	sideDir = glm::vec3(1,0,0);
	camPos = glm::vec3(0,0,-1);
	view = glm::mat4x4();
	projection = glm::mat4x4();

	this->updateView();
	this->updateProjection();
}

Camera::~Camera(void)
{
	delete &viewDir;
	delete &upDir;
	delete &sideDir;
	delete &camPos;
	delete &view;
	delete &projection;
}

void Camera::rotate(float dPhi, float dTheta)
{
	phi += dPhi;
	theta += dTheta;

	glm::mat4x4 rotX = glm::rotate(rotX, theta, glm::vec3(1, 0, 0));
	glm::mat4x4 rotY = glm::rotate(rotY, phi, glm::vec3(0, 1, 0));
	glm::mat4x4 rot = rotX * rotY;

	sideDir = glm::vec3(rot * glm::vec4(1,0,0,0));
	upDir = glm::vec3(rot * glm::vec4(0,1,0,0));
	viewDir = glm::vec3(rot * glm::vec4(0,0,1,0));

	this->updateView();
}

void Camera::move(float fb, float lr, float ud)
{
	//camPos += viewDir * fb + sideDir * lr + glm::vec3(0,1,0) * ud;
	camPos.x += (fb * viewDir.x + lr * sideDir.x);
	camPos.y += (fb * viewDir.y + lr * sideDir.y + ud);
	camPos.z += (fb * viewDir.z + lr * sideDir.z);

	this->updateView();
}

void Camera::updateView()
{
	glm::vec3 lookAt = camPos + viewDir;
	view = glm::lookAt(camPos, lookAt, upDir);
}

void Camera::updateProjection()
{
	projection = glm::perspective(90.0f, 4.0f/3.0f, 0.1f, 1000.f);//glm::frustum(-1e-2f, 1e-2f, -1e-2f, 1e-2f, 1e-2f, 100.0f);//
}

glm::mat4x4 Camera::getProjection()
{
	//this->updateProjection();
	return projection;
}

glm::mat4x4 Camera::getView()
{
	return view;
}

