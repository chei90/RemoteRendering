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


#include <Windows.h>
#include <cuda.h>
#include <cuda_runtime.h>

//Calls the Kernel for D3D Conversion
extern void callKernelD3D(int width, int height, unsigned char* yuv);
//Calls the Kernel for GL Conversion
extern void callKernelGL(int width, int height, unsigned char* yuv,  unsigned char* pixels);

//Binds D3D Texture to Array
extern "C" void bindTexture(cudaArray* array);
