//------------------------------------------------------------------------------
// File: CudaDecoder.h
// 
// Author: Ren Yifei, Lin Ziya
//
// Contact: yfren@cs.hku.hk, zlin@cs.hku.hk
// 
// Desc: The decoder class based on CUDA Decoder API. 
// It handles actual frame decoding.
//
//------------------------------------------------------------------------------

#ifndef CUDA_DECODER_H_
#define CUDA_DECODER_H_

#include "StdHeader.h"

// Auto lock for floating contexts
class CAutoCtxLock
{
private:
	CUvideoctxlock m_lock;
public:
#if USE_FLOATING_CONTEXTS
	CAutoCtxLock(CUvideoctxlock lck) { m_lock=lck; cuvidCtxLock(m_lock, 0); }
	~CAutoCtxLock() { cuvidCtxUnlock(m_lock, 0); }
#else
	CAutoCtxLock(CUvideoctxlock lck) { m_lock=lck; }
#endif
};

typedef struct
{
	CUvideoparser cuParser;
	CUvideodecoder cuDecoder;
	CUstream cuStream;
	CUvideoctxlock cuCtxLock;
	CUVIDDECODECREATEINFO dci;
	CUVIDPARSERDISPINFO DisplayQueue[DISPLAY_DELAY];
	unsigned char *pRawNV12;
	int raw_nv12_size;
	int pic_cnt;
	int display_pos;
} DecodeSession;

class DecodedStream;

class CudaH264Decoder
{
public:

	CudaH264Decoder(unsigned int maxFrame = MAX_FRM_CNT);

	virtual ~CudaH264Decoder();

	bool				Init(DecodedStream* decodedStream);

	bool				FetchVideoData(BYTE* ptr, unsigned int size);

	BYTE*				GetOutputBufferPtr() const;

protected:

	bool				InitCuda(CUvideoctxlock *pLock);

	bool				ReleaseCuda();

	static int CUDAAPI 	HandleVideoSequence(void *pvUserData, CUVIDEOFORMAT *pFormat);
	static int CUDAAPI 	HandlePictureDecode(void *pvUserData, CUVIDPICPARAMS *pPicParams);
	static int CUDAAPI 	HandlePictureDisplay(void *pvUserData, CUVIDPARSERDISPINFO *pPicParams);

	static int			PostProcessing(DecodeSession *state, CUVIDPARSERDISPINFO *pPicParams);

	static void			SendFrameDownStream();

public:

	static BYTE*		m_InputBuffer;

	static BYTE*		m_OutputYuv2Buffer;

private:

	IDirect3D9*			m_pD3D;
	IDirect3DDevice9*	m_pD3Dev;
	CUcontext			m_cuContext;
	CUdevice			m_cuDevice;
	int					m_cuInstanceCount;
	CUvideoctxlock		m_cuCtxLock;

	CUVIDPARSERPARAMS	m_parserInitParams;
	DecodeSession		m_state;

	static DecodedStream* m_DecodedStream;
};

#endif