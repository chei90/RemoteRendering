#include "UdpSocket.h"



UdpSocket::UdpSocket()
{
	WSADATA wsaData;
	int i = WSAStartup(MAKEWORD(2,2), &wsaData);
	m_Sock = -1;

	m_clientLen = sizeof(struct sockaddr_in);
}

bool UdpSocket::Create()
{
	if ((m_Sock = socket(AF_INET, SOCK_DGRAM, 0)) > 0)
		return true;
	return false;
}

void UdpSocket::setClientSocket(string address, int port)
{
	m_clientAddr.sin_family = AF_INET;
	m_clientAddr.sin_port = htons(port);
	m_clientAddr.sin_addr.s_addr = inet_addr(address.c_str());
}

struct sockaddr_in UdpSocket::getClientSocket()
{
	return m_clientAddr;
}

int UdpSocket::Receive(char *buff, int buffLen)
{
	return recvfrom(m_Sock, buff, buffLen, 0, (struct sockaddr *) &m_clientAddr, &m_clientLen);
}

int UdpSocket::Send(const char *buff, int len)
{
	return sendto(m_Sock, buff, len, 0, (struct sockaddr*) &m_clientAddr, m_clientLen);
}