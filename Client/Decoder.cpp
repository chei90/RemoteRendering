#include "Decoder.h"


Decoder::Decoder(void)
{
}

Decoder::Decoder(CUcontext &context, CUvideoctxlock &vidLock, FrameQueue* queue, int width, int height)
	:m_vidLock(vidLock), m_context(context)
{
	m_queue = queue;

	std::cout << (void*) m_queue << "Speicher Decoder" << std::endl;


	m_createFlags =  cudaVideoCreate_PreferCUVID;

	memset(&m_createInfo, 0, sizeof(CUVIDDECODECREATEINFO));
	m_createInfo.CodecType = cudaVideoCodec_H264;
	m_createInfo.ulWidth = width;
	m_createInfo.ulHeight = height;
	m_createInfo.ulNumDecodeSurfaces = 30;

	while(m_createInfo.ulNumDecodeSurfaces * m_createInfo.ulWidth * m_createInfo.ulHeight > 16 * 1024 * 1024)
		m_createInfo.ulNumDecodeSurfaces--;

	m_createInfo.ChromaFormat = cudaVideoChromaFormat_420;
	m_createInfo.OutputFormat = cudaVideoSurfaceFormat_NV12;
	m_createInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;

	m_createInfo.ulTargetWidth = m_createInfo.ulWidth;
	m_createInfo.ulTargetHeight = m_createInfo.ulHeight;
	m_createInfo.ulNumOutputSurfaces = 2;
	m_createInfo.ulCreationFlags = m_createFlags;
	m_createInfo.vidLock = vidLock;

	CUresult res = cuvidCreateDecoder(&m_decoder, &m_createInfo);

	if(res != CUDA_SUCCESS)
		std::cout << "Initialisierung des Decoders ging schief!" << std::endl;
}

Decoder::~Decoder(void)
{
	cuvidDestroyDecoder(m_decoder);
}

CUVIDDECODECREATEINFO Decoder::getDecoderParams()
{
	return m_createInfo;
}

Decoder* Decoder::getDecoder()
{
	return this;
}

CUcontext* Decoder::getContext()
{
	return &m_context;
}

FrameQueue* Decoder::getQueue()
{
	return m_queue;
}

void Decoder::initParser()
{
	CUVIDPARSERPARAMS params;
	memset(&params, 0, sizeof(CUVIDPARSERPARAMS));

	params.CodecType = m_createInfo.CodecType;
	params.ulMaxNumDecodeSurfaces = m_createInfo.ulNumDecodeSurfaces;
	params.ulMaxDisplayDelay = 1;
	params.pUserData = this;
	params.pfnSequenceCallback = HandleVideoSequence;
	params.pfnDecodePicture = HandlePictureDecode;
	params.pfnDisplayPicture = HandlePictureDisplay;

	CUresult res =  cuvidCreateVideoParser(&m_parser, &params);

	if(res != CUDA_SUCCESS) 
		std::cout << "Error at initializing the Parser!" << std::endl;
}

void Decoder::parseData(const unsigned char* data, int streamLen)
{
	CUVIDSOURCEDATAPACKET currentPic;
	memset(&currentPic, 0, sizeof(CUVIDSOURCEDATAPACKET));

	currentPic.payload_size = streamLen;
	currentPic.payload = data;

	cuvidParseVideoData(m_parser, &currentPic);
}


int CUDAAPI Decoder::HandleVideoSequence(void* userData, CUVIDEOFORMAT* format)
{
	Decoder* d = (Decoder*) userData;

	if(format->codec != d->getDecoderParams().CodecType)
		std::cout << "Format changed" << std::endl;
	return 1;
}

int CUDAAPI Decoder::HandlePictureDecode(void* userData, CUVIDPICPARAMS* params)
{
	Decoder* d = (Decoder*) userData;

	bool frameAvail = d->getQueue()->waitUntilFrameAvailable(params->CurrPicIdx);
	if(!frameAvail)
	{
		std::cout << "Too fast" << std::endl;
		return 0;
	}

	d->decodePicture(params, d->getContext());
	
	return 1;
}

int CUDAAPI Decoder::HandlePictureDisplay(void* userData, CUVIDPARSERDISPINFO* params)
{
	Decoder* d = (Decoder*) userData;
	d->getQueue()->enqueue(params);
	return 1;
}

void Decoder::decodePicture(CUVIDPICPARAMS* pictureParams, CUcontext* context)
{
	//std::cout << "Im decoding!" << std::endl;
	CUresult res = cuvidDecodePicture(m_decoder, pictureParams);
	if(res != CUDA_SUCCESS)
		std::cout << "Fehler beim Decodieren!" << std::endl;
}

void Decoder::mapFrame(int picIndex, CUdeviceptr* device, unsigned int* pitch, CUVIDPROCPARAMS* vidProcessingParams)
{
	CUresult res = cuvidMapVideoFrame(m_decoder, picIndex, device, pitch, vidProcessingParams);

	if(res != CUDA_SUCCESS)
		std::cout << "Something went wrong at mapping the Frame (VideoDecoder)" << std::endl;
}

void Decoder::unmapFrame(CUdeviceptr device)
{
	CUresult res = cuvidUnmapVideoFrame(m_decoder, device);

	if(res != CUDA_SUCCESS)
		std::cout << "Something went wrong at unmapping the Frame (VideoDecoder)" << std::endl;
}