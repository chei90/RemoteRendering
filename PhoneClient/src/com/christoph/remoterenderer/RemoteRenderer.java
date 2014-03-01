package com.christoph.remoterenderer;

import java.net.DatagramPacket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.opengl.GLES20;
import android.opengl.GLSurfaceView.Renderer;

public class RemoteRenderer implements Renderer
{
	private int m_windowWidth;
	private int m_windowHeight;
	private UdpSocket m_renderSock;
	
	public RemoteRenderer(UdpSocket socket)
	{
		super();
		this.m_renderSock = socket;
	}
	
	@Override
	public void onDrawFrame(GL10 arg0)
	{
		DatagramPacket p = m_renderSock.receive(100000);
		ByteBuffer b = ByteBuffer.wrap(p.getData());
		b.order(ByteOrder.nativeOrder());
		byte identifyer = b.get();
		if(identifyer == InputSender.FRAME_DATA)
		{
			int frameSize = b.getInt();
			System.out.println(frameSize);
			
			byte[] frameData = new byte[frameSize];
			b.get(frameData);
		}
		
		
		GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);		
	}

	@Override
	public void onSurfaceChanged(GL10 arg0, int width, int height)
	{
		m_windowHeight = height;
		m_windowWidth = width;
		// TODO Auto-generated method stub
		GLES20.glViewport(0, 0, width, height);
	}

	@Override
	public void onSurfaceCreated(GL10 arg0, EGLConfig arg1)
	{
		// Sending Client WindowSize
		ByteBuffer bBuffer = ByteBuffer.allocateDirect(64);
		bBuffer.put(InputSender.WINDOW_SIZE);
		bBuffer.putInt(800);
		bBuffer.putInt(600);
		m_renderSock.sentTo(bBuffer.array());	
		
		GLES20.glClearColor(0.0f, 1.0f, 0.0f, 1.0f);
	}
	
	public int getWindowHeight()
	{
		return m_windowHeight;
	}

	public int getWindowWidth()
	{
		return m_windowWidth;
	}
}
