package com.christoph.remoterenderer;

import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.app.Activity;
import android.view.Menu;

public class PhoneClient extends Activity
{
	
	private GLSurfaceView m_glView;

		
	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		
		m_glView = new RemoteRenderingSurfaceView(this);
		setContentView(m_glView);
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu)
	{
		// Inflate the menu; this adds items to the action bar if it is present.
		getMenuInflater().inflate(R.menu.phone_client, menu);
		return true;
	}

}
