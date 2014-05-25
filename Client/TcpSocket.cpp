#include "TcpSocket.h"



TcpSocket::TcpSocket()
{
	WSADATA wsaData;

	int i = WSAStartup(MAKEWORD(2,2), &wsaData);

	m_Sock = -1;
}

bool TcpSocket::Create()
{
	if ((m_Sock = socket(AF_INET, SOCK_STREAM, 0)) > 0)
		return true;
	return false;
}

bool TcpSocket::Listen(int que)
{
	if (listen(m_Sock, que) == 0)
		return true;
	return false;
}

bool TcpSocket::Accept(TcpSocket &clientSock)
{
	int size = sizeof(struct sockaddr);
	printf("Accepting\n");
	clientSock.m_Sock = accept(m_Sock,
		(struct sockaddr *) &clientSock.m_SockAddr, (socklen_t *) &size);
	if (clientSock.m_Sock == -1)
		cout << "accept failed: " << WSAGetLastError() << endl;
	WSACleanup();
	return false;
	return true;
}

bool TcpSocket::Connect(string address, int port)
{
	struct in_addr *addr_ptr;
	struct hostent *hostPtr;
	string add;
	try
	{
		hostPtr = gethostbyname(address.c_str());
		if (hostPtr == NULL)
			return false;

		// the first address in the list of host addresses
		addr_ptr = (struct in_addr *) *hostPtr->h_addr_list;

		// changed the address format to the Internet address in standard dot notation
		add = inet_ntoa(*addr_ptr);
		if (add.c_str() == "")
			return false;
	} catch (int e)
	{
		return false;
	}

	struct sockaddr_in sockAddr;
	sockAddr.sin_family = AF_INET;
	sockAddr.sin_port = htons(port);
	sockAddr.sin_addr.s_addr = inet_addr(add.c_str());
	if (connect(m_Sock, (struct sockaddr *) &sockAddr, sizeof(struct sockaddr))== 0)
		return true;
	return false;

}

int TcpSocket::Receive(char *buff, int buffLen)
{
	return recv(m_Sock, buff, buffLen, 0);
}

int TcpSocket::Send(const char *buff, int len)
{
	return send(m_Sock, buff, len, 0);
}