#ifndef RENDERSOCKET_H_
#define RENDERSOCKET_H_

#include <iostream>
#include <WinSock2.h>
#include <ws2tcpip.h>
#include <Windows.h>
#include "Magicnumbers.h"

// Need to link with Ws2_32.lib
#pragma comment (lib, "Ws2_32.lib")

using std::string;


typedef
	int socklen_t;

using std::cout;
using std::endl;

class UdpSocket{

private: 
	int m_Sock;
	int m_clientLen;
	struct sockaddr_in m_SockAddr;
	struct sockaddr_in m_clientAddr;

public:
	UdpSocket();
	bool Create();
	void SetToNonBlock();
	bool Bind(string address, int port);
	int Receive(char *buff, int buffSize);
	int Send(const char *buff, int len);
	void setClientSocket(struct sockaddr_in &client);
	struct sockaddr_in getClientSocket();
	bool Close();
};

#endif