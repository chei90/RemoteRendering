#version 150 
#define NR_LIGHTS 4 
precision highp float;
//Light uniforms
uniform vec3 uLightPosition[NR_LIGHTS];
uniform vec3  uLa[NR_LIGHTS];   //ambient
uniform vec3  uLd[NR_LIGHTS];   //diffuse
uniform vec3  uLs[NR_LIGHTS];   //specular


//Material uniforms
uniform vec3  uKa;   //ambient
uniform vec3  uKd;   //diffuse
uniform vec3  uKs;   //specular
uniform float uNs;   //specular coefficient
uniform float d;     //Opacity

in vec3 vNormal;
in vec3 vLightRay[NR_LIGHTS];
in vec3 vEye[NR_LIGHTS];

void main(void) {
  
   vec3 COLOR = vec3(0.0,0.0,0.0);
   vec3 N =  normalize(vNormal);
   vec3 L =  vec3(0.0,0.0,0.0);
   vec3 E =  vec3(0.0,0.0,0.0);
   vec3 R =  vec3(0.0,0.0,0.0);
   vec3 deltaRay = vec3(0.0);
   const int  lsize = 2;
   const float step = 0.25;
   const float inv_total = 1.0/((float(lsize*lsize) + 1.0)*(float(lsize*lsize) + 1.0));  //how many deltaRays
  
   for(int i = 0; i < NR_LIGHTS; i++){
		E = normalize(vEye[i]);
		L = normalize(vLightRay[i]);
		R = reflect(L, N);
		COLOR += (uLa[i] * uKa);
		COLOR += (uLd[i] * uKd * clamp(dot(N,-L),0.0,1.0));
		COLOR += (uLs[i] * uKs * pow( max(dot(R, E), 0.0), uNs) * 4.0);    
   }
		
   gl_FragColor =  vec4(COLOR, d);
   return;
   
}

    