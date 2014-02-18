package com.christoph.remoterenderer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.opengl.GLES20;
import android.opengl.GLSurfaceView.Renderer;

public class RemoteRenderer implements Renderer
{
	private int m_windowWidth;
	private int m_windowHeight;
	private int m_program;
	private ScreenQuad m_quad;
	
	
	private final String vertexShaderCode =
		    "attribute vec4 vPosition;" +
		    "void main() {" +
		    "  gl_Position = vPosition;" +
		    "}";

	private final String fragmentShaderCode =
		    "precision mediump float;" +
		    "uniform vec4 vColor;" +
		    "void main() {" +
		    "  gl_FragColor = vColor;" +
		    "}";
	
	public static int loadShader(int type, String shaderCode)
	{
		int shader = GLES20.glCreateShader(type);

		GLES20.glShaderSource(shader, shaderCode);
		GLES20.glCompileShader(shader);

		return shader;
	}
	
	
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
		
		m_quad = new ScreenQuad(400,400);
		
		int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderCode);
		int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderCode);
		
		m_program = GLES20.glCreateProgram();
		GLES20.glAttachShader(m_program, fragmentShader);
		GLES20.glAttachShader(m_program, vertexShader);
		GLES20.glLinkProgram(m_program);
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
