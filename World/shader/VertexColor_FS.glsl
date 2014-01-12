#version 150 core

in vec2 fs_in_tc;
uniform sampler2D colorTex;

out vec4 fs_out_color;

void main(void) {
    //fs_out_color = vec4(fs_in_color, 1);
    fs_out_color = texture(colorTex, fs_in_tc);
}