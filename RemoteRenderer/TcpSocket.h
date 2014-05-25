#ifndef TCPSOCKET_H_
#define TCPSOCKET_H_

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

class TcpSocket : public Socket{

public:
	TcpSocket();
	bool Create();
	bool Listen(int que);
	bool Accept(TcpSocket &clientSock);
	bool Connect(string address, int port);
	int Receive(char *buff, int buffSize);
	int Send(const char *buff, int len);
};

#endif