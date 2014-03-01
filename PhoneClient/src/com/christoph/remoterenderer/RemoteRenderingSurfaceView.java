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
		
		Thread t = new Thread(new Runnable()
		{
			public void run()
			{
				m_renderSock = new UdpSocket("192.168.178.42", 8080, "192.168.178.50", 8081);
				byte [] msg = new String("Hi").getBytes();
				m_renderSock.sentTo(msg);	
				m_renderSock.close();
			}
		});
		t.start();
		
		m_renderer = new RemoteRenderer();
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
