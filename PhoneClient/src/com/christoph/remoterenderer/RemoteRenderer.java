package com.christoph.remoterenderer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.opengl.GLES20;
import android.opengl.GLSurfaceView.Renderer;

public class RemoteRenderer implements Renderer
{
	private int m_windowWidth;
	private int m_windowHeight;
	private UdpSocket m_renderSock;

	
	@Override
	public void onDrawFrame(GL10 arg0)
	{
		// TODO Auto-generated method stub
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
		// TODO Auto-generated method stub
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

	public void addUdpSocket(UdpSocket m_renderSock)
	{
		this.m_renderSock = m_renderSock;		
	}
}
