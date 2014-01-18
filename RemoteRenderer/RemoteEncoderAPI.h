#pragma once

#define CM_API __stdcall

#ifdef __COMPILING_DLL
#define CM_DLL_API __declspec(dllexport)
#else 
#define CM_DLL_API __declspec(dllimport)
#endif

#include <string>

typedef void (*KeyBoardHandler)(int key, bool pressed);
typedef void (*MouseHandler)(int dx, int dy, int button);

enum GFX_API
{
	D3D,
	GL
};

struct RREncoderDesc
{
	GFX_API gfxapi;
	unsigned int w;
	unsigned int h;
	const char* ip;
	int port;
	KeyBoardHandler keyHandler;
	MouseHandler mouseHandler;
};

CM_DLL_API bool CM_API RRInit(RREncoderDesc& desc);

CM_DLL_API void CM_API RRSetSource(void* ptr);

CM_DLL_API const RREncoderDesc& RRGetDesc(void);

CM_DLL_API bool CM_API RRDelete(void);

CM_DLL_API void CM_API RREncode();

CM_DLL_API void CM_API RRWaitForConnection();

CM_DLL_API void CM_API RRQueryClientEvents();