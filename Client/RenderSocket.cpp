#include "RenderSocket.h"



RenderSocket::RenderSocket()
{
	WSADATA wsaData;

	int i = WSAStartup(MAKEWORD(2,2), &wsaData);

	m_Sock = -1;
}

bool RenderSocket::Create()
{
	if ((m_Sock = socket(AF_INET, SOCK_STREAM, 0)) > 0)
		return true;
	return false;
}

void RenderSocket::SetToNonBlock()
{
	u_long iMode=1;
	ioctlsocket(m_Sock,FIONBIO,&iMode);
}

bool RenderSocket::Bind(string address, int port)
{
	m_SockAddr.sin_family = AF_INET;
	m_SockAddr.sin_port = htons(port);
	m_SockAddr.sin_addr.s_addr = inet_addr(address.c_str());

	if (bind(m_Sock, (struct sockaddr *) &m_SockAddr, sizeof(struct sockaddr))
		== 0)
		return true;
	return false;
}

bool RenderSocket::Listen(int que)
{
	if (listen(m_Sock, que) == 0)
		return true;
	return false;
}

bool RenderSocket::Close()
{
	if (m_Sock > 0)
	{
		closesocket(m_Sock);
		WSACleanup();
		m_Sock = 0;
		return true;
	}
	return false;

}
bool RenderSocket::Accept(RenderSocket &clientSock)
{
	int size = sizeof(struct sockaddr);
	clientSock.m_Sock = accept(m_Sock,
		(struct sockaddr *) &clientSock.m_SockAddr, (socklen_t *) &size);
	if (clientSock.m_Sock == -1)
		cout << "accept failed: " << WSAGetLastError() << endl;

	WSACleanup();
	return false;
	return true;
}

bool RenderSocket::Connect(string address, int port)
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

int RenderSocket::Receive(char *buff, int buffLen)
{
	return recv(m_Sock, buff, buffLen, 0);
}

int RenderSocket::Send(const char *buff, int len)
{
	return send(m_Sock, buff, len, 0);
}