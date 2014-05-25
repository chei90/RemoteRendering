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

