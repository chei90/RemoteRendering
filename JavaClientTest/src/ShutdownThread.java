
public class ShutdownThread extends Thread
{
	private UdpSocket m_sock;
	
	public ShutdownThread(UdpSocket sock)
	{
		this.m_sock = sock;
	}
	
	@Override
	public void run()
	{
		System.out.println("shutting down");
		m_sock.close();
	}
	
}
