package com.christoph.remoterenderer;

import java.nio.ByteBuffer;

public class InputSender implements Runnable
{
	static final byte KEY_PRESSED = 1;
	static final byte KEY_RELEASED = 2;
	static final byte SPECIAL_KEY_PRESSED = 3;
	static final byte SPECIAL_KEY_RELEASED = 4;
	static final byte SHUTDOWN_CONNECTION = 5;
	static final byte FRAME_DATA = 6;
	static final byte WINDOW_SIZE = 7;
	static final byte MOUSE_PRESSED = 8;
	static final byte MOUSE_RELEASED = 9;

	
	private boolean m_L;
	private boolean m_R;
	private boolean m_O;
	private boolean m_I;
	
	private ByteBuffer msg;
	
	private UdpSocket m_renderSock;
	
	public InputSender(UdpSocket socket)
	{
		this.m_renderSock = socket;
		msg = ByteBuffer.allocateDirect(64);
	}
	
	public void moveLeft()
	{
		m_L = true;
	}
	
	public void moveRight()
	{
		m_R = true;
	}
	
	public void moveIn()
	{
		m_I = true;
	}
	
	public void moveOut()
	{
		m_O = true;
	}
	
	public void release()
	{
		m_L = m_R = m_I = m_O = false;
	}
	
	@Override
	public void run()
	{
		while(true)
		{
			
		}
	}
	
}
