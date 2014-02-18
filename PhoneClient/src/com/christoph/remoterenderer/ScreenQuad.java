package com.christoph.remoterenderer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

import android.opengl.GLES20;

public class ScreenQuad
{
	
	private FloatBuffer vertices;
	private ShortBuffer indices;
	private int m_program;
	
	static final byte COORDS_PER_VERTEX = 3;
	static float squareCoords [] = 
		{0f, 1f, 0f,
		 0f, 0f, 0f,
		 1f, 0f, 0f,
		 1f, 1f, 0f};
	
	private short drawOrder[] = {0, 1, 2, 0, 2, 3};
	
	public ScreenQuad(int width, int height)
	{
		squareCoords[1] *= height; squareCoords[0] *= width;
		squareCoords[4] *= height; squareCoords[3] *= width;
		squareCoords[7] *= height; squareCoords[6] *= width;
		squareCoords[10] *= height; squareCoords[9] *= width;
		
		ByteBuffer buffer = ByteBuffer.allocateDirect(squareCoords.length * 4);
		buffer.order(ByteOrder.nativeOrder());
		vertices =  buffer.asFloatBuffer();
		vertices.put(squareCoords);
		vertices.position(0);
		
		ByteBuffer dlb = ByteBuffer.allocateDirect(drawOrder.length * 2);
		dlb.order(ByteOrder.nativeOrder());
		indices = dlb.asShortBuffer();
		indices.put(drawOrder);
		indices.position(0);
	}
	
	public void draw()
	{
		GLES20.glUseProgram(m_program);
		int posHandle = GLES20.glGetAttribLocation(m_program, "vPosition");
		GLES20.glEnableVertexAttribArray(posHandle);
		GLES20.glVertexAttribPointer(posHandle, COORDS_PER_VERTEX, GLES20.GL_FLOAT,
				false, vertexStride, vertices);
	}
}
