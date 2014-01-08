#version 150 core

uniform mat4 model;
uniform mat4 viewProj;

in vec3 vs_in_pos;
in vec3 vs_in_color;

out vec3 fs_in_color;

void main(void) {
    gl_Position = viewProj * model * vec4(vs_in_pos, 1);
	fs_in_color = vs_in_color;
}