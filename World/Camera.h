#pragma once

#include <gtc\matrix_inverse.hpp>
#include <gtc\matrix_transform.hpp>
#include <gtc\type_ptr.hpp>


class Camera
{
public:
	Camera(void);
	~Camera(void);

	void rotate(float dPhi, float dTheta);
	void move(float fb, float lr, float ud);
	void updateView();
	void updateProjection();
	glm::mat4x4 getProjection();
	glm::mat4x4 getView();

private:
	float phi;
	float theta;
	glm::vec3 viewDir;
	glm::vec3 upDir;
	glm::vec3 sideDir;
	glm::vec3 camPos;
	glm::mat4x4 view;
	glm::mat4x4 projection;
};

