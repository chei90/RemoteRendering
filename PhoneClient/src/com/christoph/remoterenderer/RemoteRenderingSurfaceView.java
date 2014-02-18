package com.christoph.remoterenderer;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.view.MotionEvent;

public class RemoteRenderingSurfaceView extends GLSurfaceView
{
	
	public RemoteRenderingSurfaceView(Context context)
	{
		super(context);
		
		setEGLContextClientVersion(2);
		setRenderer(new RemoteRenderer());
	}
	
	@Override
	public boolean onTouchEvent(MotionEvent e)
	{
		System.out.println(e.getX());
		
		return true;
	}
}
