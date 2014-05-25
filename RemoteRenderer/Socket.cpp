#include "Socket.h"


Socket::Socket(void)
{
	WSADATA wsaData;

	int i = WSAStartup(MAKEWORD(2,2), &wsaData);

	m_Sock = -1;
}

void Socket::SetToNonBlock()
{
	u_long iMode=1;
	ioctlsocket(m_Sock,FIONBIO,&iMode);
}

bool Socket::Bind(string address, int port)
{
	m_SockAddr.sin_family = AF_INET;
	m_SockAddr.sin_port = htons(port);
	m_SockAddr.sin_addr.s_addr = inet_addr(address.c_str());

	if (bind(m_Sock, (struct sockaddr *) &m_SockAddr, sizeof(struct sockaddr))
		== 0)
		return true;
	return false;
}

bool Socket::Close()
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


Socket::~Socket(void)
{
}
