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

