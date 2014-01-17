#pragma once

#include "RemoteRendererAPI.h"
#include <iostream>
#pragma comment(lib, "Server.lib")

void doNothing(int key, bool pressed)
{
	std::cout << "Hi!" << std::endl;
}

int main()
{
	RREncoderDesc rdesc;
	rdesc.gfxapi = GL;
	rdesc.w = 800;
	rdesc.h = 600;
	rdesc.ip = "127.0.0.1";
	rdesc.port = 8081;
	rdesc.keyHandler = doNothing;

	RRInit(rdesc);
	RRWaitForConnection();

	while(true)
	{
		RRQueryClientEvents();
	}
}