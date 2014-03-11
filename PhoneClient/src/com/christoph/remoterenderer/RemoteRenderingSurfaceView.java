package com.christoph.remoterenderer;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.view.MotionEvent;

public class RemoteRenderingSurfaceView extends GLSurfaceView
{
	private UdpSocket m_renderSock;
	private RemoteRenderer m_renderer;
	
	public RemoteRenderingSurfaceView(Context context)
	{
		super(context);
		
		setEGLContextClientVersion(2);
		m_renderSock = new UdpSocket("192.168.178.45", 8080, "192.168.178.50", 8081);
		m_renderer = new RemoteRenderer(m_renderSock, this);
		setRenderer(m_renderer);
		Runtime.getRuntime().addShutdownHook(new ShutdownThread(m_renderSock));		
	}
	
	@Override
	public boolean onTouchEvent(MotionEvent e)
	{
		System.out.println(e.getX());
		return true;
	}
}
