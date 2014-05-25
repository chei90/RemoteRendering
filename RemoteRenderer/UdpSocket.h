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
