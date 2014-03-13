package com.christoph.remoterenderer;

import java.net.DatagramPacket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.media.MediaCodec;
import android.media.MediaCodec.BufferInfo;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public class PhoneClient extends Activity implements SurfaceHolder.Callback
{
	private DisplayThread remoteRenderer = null;
	private int m_windowWidth;
	private int m_windowHeight;
	
	@Override
	protected void onCreate(Bundle savedInstanceState)
	{
		super.onCreate(savedInstanceState);
		SurfaceView sv = new SurfaceView(this);
		sv.getHolder().addCallback(this);
		setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
		setContentView(sv);
	}

	protected void onDestroy()
	{
		super.onDestroy();
	}

	@Override
	public void surfaceCreated(SurfaceHolder holder)
	{
	}

	@Override
	public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
	{
		m_windowWidth = width;
		m_windowHeight = height;
		
		System.out.println("Width: " + width + " Height: " + height);
		
		if (remoteRenderer == null)
		{
			remoteRenderer = new DisplayThread(holder.getSurface());
			remoteRenderer.start();
		}
	}

	@Override
	public void surfaceDestroyed(SurfaceHolder holder)
	{
		if (remoteRenderer != null)
		{
			remoteRenderer.interrupt();
		}
	}

	private class DisplayThread extends Thread
	{
		private MediaCodec codec;
		private Surface surface;
		private UdpSocket m_renderSock;
		private ByteBuffer [] codecInputBuffers;
		private ByteBuffer [] codecOutputBuffers;
		
		public DisplayThread(Surface surface)
		{
			this.surface = surface;
		}

		@Override
		public void run()
		{
			m_renderSock = new UdpSocket("192.168.178.45", 8080, "192.168.178.50", 8081);
			
			ByteBuffer bBuffer = ByteBuffer.allocateDirect(64);
			bBuffer.put(MagicNumbers.WINDOW_SIZE);
			bBuffer.putInt(800);
			bBuffer.putInt(600);
			m_renderSock.sentTo(bBuffer.array());
			
			
			//Configuring Media Decoder
			codec = MediaCodec.createDecoderByType("video/avc");
			
			MediaFormat format = new MediaFormat();
			format.setString(MediaFormat.KEY_MIME, "video/avc");
			format.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 100000);
			format.setInteger(MediaFormat.KEY_WIDTH, 800);
			format.setInteger(MediaFormat.KEY_HEIGHT, 600);
			
			codec.configure(format, surface, null, 0);
			codec.start();
			
			codecInputBuffers = codec.getInputBuffers();
			codecOutputBuffers = codec.getOutputBuffers();
			
			while(!Thread.interrupted())
			{
				int frameSize = 0;
				byte[] frameData = null;
				
				//Receiving
				DatagramPacket p = m_renderSock.receive(100000);
				ByteBuffer b = ByteBuffer.wrap(p.getData());
				b.order(ByteOrder.nativeOrder());
				byte identifyer = b.get();
				
				if(identifyer == MagicNumbers.FRAME_DATA)
				{
					frameSize = b.getInt();
					
					frameData = new byte[frameSize];
					b.get(frameData);
				}
				
				
				int inIndex = codec.dequeueInputBuffer(10000);
				if(inIndex >= 0)
				{
					ByteBuffer inputBuffer = codecInputBuffers[inIndex];
					inputBuffer.clear();
					inputBuffer.put(frameData);
					
					codec.queueInputBuffer(inIndex, 0, frameSize, 33, 0);
				}
				
				BufferInfo buffInfo = new MediaCodec.BufferInfo();
				int outIndex = codec.dequeueOutputBuffer(buffInfo, 10000);
	
				switch(outIndex)
				{
				case MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED:
					codecOutputBuffers = codec.getOutputBuffers();
					System.out.println("OB Changed");
					break;
				case MediaCodec.INFO_OUTPUT_FORMAT_CHANGED:
					System.out.println("OF Changed");
					break;
				case MediaCodec.INFO_TRY_AGAIN_LATER:
					System.out.println("l8r");
					break;
				default:
					ByteBuffer buffer = codecOutputBuffers[outIndex];
					codec.releaseOutputBuffer(outIndex, true);
				}
			}
			
			m_renderSock.close();
		}
	}
}