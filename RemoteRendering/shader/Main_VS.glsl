#version 150 core


in vec3 vs_in_pos; 
in vec3 vs_in_normal;


// Licht:
uniform vec3 uAmbientColor; 
uniform vec3 uLightingDirection; 
uniform vec3 uDirectionalColor; 
uniform mat4 uMVMatrix;
uniform mat4 uPMatrix;
uniform mat4 uNMatrix;

out vec3 vLightWeighting; 

void main(void) 
{
            gl_Position = uPMatrix  * uMVMatrix * vec4(vs_in_pos, 1.0); 
            //BEGINN BELEUCHTUNG 
            vec4 transformedNormal = uNMatrix * vec4(vs_in_normal, 1.0); 
            float fDirectionalLightWeighting = max(dot(transformedNormal.xyz, -uLightingDirection), 0.0);
            vLightWeighting = uAmbientColor + uDirectionalColor * fDirectionalLightWeighting;
            //ENDE BELEUCHTUNG
         } 
    