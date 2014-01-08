#ifndef DECODER_H_
#define DECODER_H_

#include <windows.h>
#include <cuda.h>
#include <cuviddec.h>
#include <nvcuvid.h>

#include <iostream>

#include "FrameQueue.h"

class Decoder
{
public:
	//Konstruktoren
	Decoder(void);
	Decoder(CUcontext &context, CUvideoctxlock &vidLock, FrameQueue* queue, int width, int heigt);
	~Decoder(void);

	//Get-Set
	CUVIDDECODECREATEINFO getDecoderParams();
	Decoder* getDecoder();
	CUcontext* getContext();
	FrameQueue* getQueue();

	//Parser
	void initParser();
	void parseData(const unsigned char* stream, int streamLen);

	//Decoding Stuff
	void decodePicture(CUVIDPICPARAMS* pictureParams, CUcontext* context);
	void mapFrame(int picIndex, CUdeviceptr* device, unsigned int* pitch, CUVIDPROCPARAMS* vidProcessingParams);
	void unmapFrame(CUdeviceptr device);




private:
	CUvideoctxlock m_vidLock;
	CUcontext m_context;
	CUVIDDECODECREATEINFO m_createInfo;
	cudaVideoCreateFlags m_createFlags;
	CUvideodecoder m_decoder;
	CUvideoparser m_parser;
	CUVIDPARSERDISPINFO* m_currentFrame;
	FrameQueue* m_queue;


	static int CUDAAPI HandleVideoSequence(void* userData, CUVIDEOFORMAT* format);
	static int CUDAAPI HandlePictureDecode(void* userData, CUVIDPICPARAMS* params);
	static int CUDAAPI HandlePictureDisplay(void* userData, CUVIDPARSERDISPINFO* params);

};

#endif