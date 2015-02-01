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

#include <iostream>
#include <WinSock2.h>
#include <ws2tcpip.h>
#include <Windows.h>
#include "Magicnumbers.h"
#include "Socket.h"

// Need to link with Ws2_32.lib
#pragma comment (lib, "Ws2_32.lib")

using std::string;


typedef
	int socklen_t;

using std::cout;
using std::endl;

class UdpSocket : public Socket{

private: 
	int m_clientLen;
	struct sockaddr_in m_clientAddr;

public:
	UdpSocket();
	bool Create();
	int Receive(char *buff, int buffSize);
	int Send(const char *buff, int len);
	void setClientSocket(string address, int port);
	struct sockaddr_in getClientSocket();
};
