#version 150 core

const vec4 color = vec4(0.0, 0.0, 1.0, 1.0); // blue

in float brightness;

out vec4 fs_out_color;

void main(void) {
    fs_out_color = brightness * color;
}