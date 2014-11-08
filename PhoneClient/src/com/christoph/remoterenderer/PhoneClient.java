package com.christoph.remoterenderer;

import java.net.DatagramPacket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.media.MediaCodec;
import android.media.MediaCodec.BufferInfo;
import android.media.MediaFormat;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public class PhoneClient extends Activity implements SurfaceHolder.Callback
{
	private DisplayThread remoteRenderer = null;
	
	protected int touchId = -1;
	protected final int TOUCHED = 1;
	protected final int RELEASED = 2;
	
	//Steuerung
	private boolean left = false, tmpLeft = false, 
			right = false, tmpRight = false;
	
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
	
	@Override
	public boolean onTouchEvent(MotionEvent e)
	{
		if(e.getAction() == MotionEvent.ACTION_DOWN)
		{
			touchId = TOUCHED;
		}
		if(e.getAction() == MotionEvent.ACTION_UP)
		{
			touchId = RELEASED;
		}
		
		if(e.getX() <= 300)
			left = true;
		else if(e.getX() <= 600)
			right = true;
		
		return true;
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
			m_renderSock = new UdpSocket("192.168.178.20", 8080, "192.168.178.50", 8081);
			
			ByteBuffer bBuffer = ByteBuffer.allocateDirect(64);
			bBuffer.put(MagicNumbers.WINDOW_SIZE);
			bBuffer.putInt(960);
			bBuffer.putInt(540);
			m_renderSock.sentTo(bBuffer.array());
			
			
			//Configuring Media Decoder
			codec = MediaCodec.createDecoderByType("video/avc");
			
			MediaFormat format = new MediaFormat();
			format.setString(MediaFormat.KEY_MIME, "video/avc");
			format.setInteger(MediaFormat.KEY_MAX_INPUT_SIZE, 100000);
			format.setInteger(MediaFormat.KEY_WIDTH, 960);
			format.setInteger(MediaFormat.KEY_HEIGHT, 540);
			
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
					break;
				case MediaCodec.INFO_OUTPUT_FORMAT_CHANGED:
					break;
				case MediaCodec.INFO_TRY_AGAIN_LATER:
					break;
				default:
					ByteBuffer buffer = codecOutputBuffers[outIndex];
					codec.releaseOutputBuffer(outIndex, true);
				}
				
				if(touchId >= 0)
				{
					ByteBuffer keyMsg = ByteBuffer.allocateDirect(64);
					
					if(touchId == TOUCHED)
						keyMsg.put(MagicNumbers.KEY_PRESSED);
					if(touchId == RELEASED)
						keyMsg.put(MagicNumbers.KEY_RELEASED);
					
					if(left)
					{
						keyMsg.put((byte) 'q');
						left = false;
					} else
					if(right)
					{
						keyMsg.put((byte) 'e');
						right = false;
					}
					else
						keyMsg.put((byte) 'w');
					
					m_renderSock.sentTo(keyMsg.array());
					
					touchId = -1;
				}
			}
			
			m_renderSock.close();
		}
	}
}