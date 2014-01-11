#version 150 core

uniform sampler2D colorTex;
in vec3 fs_in_color;
in vec2 fs_in_texCoods;
out vec4 fs_out_color;

void main(void) {
    //fs_out_color = vec4(texture(colorTex, fs_in_texCoords),1);
	fs_out_color = vec4(fs_in_color, 1);
}