#pragma once

#define CM_API __stdcall

#ifdef __COMPILING_DLL
#define CM_DLL_API __declspec(dllexport)
#else 
#define CM_DLL_API __declspec(dllimport)
#endif

#include <string>

/**
 * \brief API Callback to handle received UserInput    
 * \param[in] key Indentifyes which ASCII Key was pressed,
 *					eg. 'w' = 119
 * \param[in] pressed Whether key was pressed or Released
 * 
 * \remarks Callback is only called once on KeyHit and KeyRelease
*/
typedef void (*KeyBoardHandler)(int key, bool pressed);

/**
 * \brief API Callback to handle MouseMovement
 *			Currently ONLY Left Mousebutton is supported
 * \param[in] dx Mousemovement during last frame in X-Dir
 * \param[in] dy Mousemovement during last frame in Y-Dir
 * \param[in] button Pressed Mousebutton currently <b> only </b>
 *			Left Mousebutton is supported and set by default   
 * \param[in] state States if Mousebutton was pressed (state = 1)
 *          or released
 * \remarks MouseMovement is only broadcasted and processed if 
 *			MouseKey is pressed                                                               
*/
typedef void (*MouseHandler)(int dx, int dy, int button, int state);

enum GFX_API
{
	D3D,
	GL
};

/**
 * Struct holding all neccessary Information to setup RemoteEncoderAPI
 * \param GFX_API GL or D3D Graphics
 * \param w has to be set to 800, no other Res currently supported
 * \param h has to be set to 600, no other Res currently supported
 * \param ip the IP you want to bind the server to, has to be localHost or
 *				the current machine ip (eg. cmd -> ipconfig)
 * \param port the port you want to use
 * \param KeyBoardHandler KeyBoard callback
 * \param MouseHandler MouseCallback
*/
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

/**
 * \brief Initializes an Instance of RemoteEncoder   
 * \param[in] desc neccessary parameters to init RemoteEncoder
 * \return true if successful                                                                
*/
CM_DLL_API bool CM_API RRInit(RREncoderDesc& desc);

/**
 * \brief Sets OpenGL Source
 * \param[in] ptr has to be OpenGL PixelBufferObject or ID3D11Resource*
*/
CM_DLL_API void CM_API RRSetSource(void* ptr);

/**
 * \brief Returns RRDesc   
 * \return Description of current Instance                         
*/
CM_DLL_API const RREncoderDesc& RRGetDesc(void);

/**
 * \brief Deletes current Instance of RR   
 * \return true if successfully deleted                            
*/
CM_DLL_API bool CM_API RRDelete(void);

/**
 * \brief Encodes current GL frame                                    
*/
CM_DLL_API void CM_API RREncode();

/**
 * \brief Waits for Client to connect   
 *			As soon as a Client connects to previously bound 
 *          IP and Port, the API sends encoded frames to
 *			this IP
*/
CM_DLL_API void CM_API RRWaitForConnection();

/**
 * \brief Queries Client for sent UserInput   
 *			Calls API Callbacks if Input was sent                  
*/
CM_DLL_API void CM_API RRQueryClientEvents();

/**
 *                                                                      
*/
CM_DLL_API bool CM_API RRIsKeyDown(char key);