#include "Camera.h"


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

	glm::vec4 t_side = rot * glm::vec4(1,0,0,0);
	sideDir.x = t_side.x;
	sideDir.y = t_side.y;
	sideDir.z = t_side.z;
	delete &t_side;

	glm::vec4 t_up = rot * glm::vec4(0,1,0,0);
	upDir.x = t_up.x;
	upDir.y = t_up.y;
	upDir.z = t_up.z;
	delete &t_up;

	glm::vec4 t_view = rot * glm::vec4(0,0,1,0);
	viewDir.x = t_view.x;
	viewDir.y = t_view.y;
	viewDir.z = t_view.z;
	delete &t_view;
}

void Camera::move(float fb, float lr, float ud)
{
	camPos.x += (fb * viewDir.x + lr * sideDir.x);
	camPos.y += (fb * viewDir.y + lr * sideDir.y + ud);
	camPos.z += (fb * viewDir.z + lr * sideDir.z);
}

void Camera::updateView()
{
	glm::vec3 lookAt = camPos + viewDir;
	glm::lookAt(camPos, lookAt, upDir);
}

void Camera::updateProjection()
{
	projection = glm::frustum(-1e-2f, 1e-2f, -1e-2f, 1e-2f, 1e-2f, 1e+2f);
}

glm::mat4x4 Camera::getProjection()
{
	this->updateProjection();
	return projection;
}

glm::mat4x4 Camera::getView()
{
	this->updateView();
	return view;
}

