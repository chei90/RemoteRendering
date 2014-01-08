#version 150
in vec3 vs_in_pos;
//in vec3 vs_in_normal;
//in vec4 vertexColor;

uniform mat4 modelView;
uniform mat4 proj;
//uniform mat4 normals;
//const int NR_LIGHTS = 4;
//uniform vec3 uLightPosition[NR_LIGHTS];

// out-Variable der Farbe f�r den Fragment-Shader

//out vec3 vNormal;
//out vec3 vLightRay[NR_LIGHTS];
//out vec3 vEye[NR_LIGHTS];

void main(void) {

	 vec4 vertex = modelView * vec4(vs_in_pos, 1.0);
	 //vNormal = vec3(normals * vec4(vs_in_normal, 1.0));
	 //vec4 lightPosition = vec4(0.0);
	 
	 //for(int i = 0; i < NR_LIGHTS; i++){
	 //	lightPosition = modelView * vec4(uLightPosition[i], 1.0);
	 //	vLightRay[i] = vertex.xyz - lightPosition.xyz;
	 //	vEye[i] = -vec3(vertex.xyz);
	 //}

	 gl_Position = proj * modelView * vec4(vs_in_pos, 1.0);
	 
}  