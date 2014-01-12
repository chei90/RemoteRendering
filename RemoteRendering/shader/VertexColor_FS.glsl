#version 150 core

uniform sampler2D colorTex;
in vec3 fs_in_color;
in vec2 fs_in_texCoords;
out vec4 fs_out_color;

void main(void) {
    //float b = 0.5 + 0.5 * dot(fs_in_color, vec3(0,1,0));
    fs_out_color = texture(colorTex, fs_in_texCoords);
	//fs_out_color = vec4(fs_in_color, 1);
}