import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.SocketException;
import java.net.UnknownHostException;

public class UdpSocket
{
	protected DatagramSocket m_sock;
	protected InetAddress serverIp;
	protected int serverPort;
	
	public UdpSocket(String ip, int port, String sIp, int sPort)
	{
		try
		{
			m_sock = new DatagramSocket(new InetSocketAddress(ip, port));
		} catch (SocketException e)
		{
			System.err.println("Failed creating ClientPort");
			e.printStackTrace();
		}
		
		try
		{
			serverIp = InetAddress.getByName(sIp);
			serverPort = sPort;
		} catch (UnknownHostException e)
		{
			System.err.println("Failed creating ServerAddress");
			e.printStackTrace();
		}
		
	}
	
	public boolean sentTo(byte [] msg)
	{
		DatagramPacket packet = new DatagramPacket(msg, msg.length, serverIp, serverPort);
		try
		{
			m_sock.send(packet);
		} catch (IOException e)
		{
			return false;
		}
		return true;
	}
	
	public DatagramPacket receive(int size)
	{
		DatagramPacket packet = new DatagramPacket(new byte[size], size);
		try
		{
			m_sock.receive(packet);
		} catch (IOException e)
		{
			return null;
		}
		return packet;
	}
	
	public void close()
	{
		m_sock.close();
	}
}
