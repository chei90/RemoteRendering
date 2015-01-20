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


using std::string;

class Socket
{

public:
	Socket(void);
	~Socket(void);

	virtual bool Create() = 0;
	void SetToNonBlock();
	bool Bind(string address, int port);
	virtual int Receive(char *buff, int buffSize) = 0;
	virtual int Send(const char *buff, int len) = 0;
	bool Close();

protected:
	int m_Sock;
	struct sockaddr_in m_SockAddr;
};

