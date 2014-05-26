#include "RemoteEncoder.h"
#include <fstream>
/************************************************************************/
/* Callbacks                                                            */
/************************************************************************/
static unsigned char* __stdcall HandleAquireBitStream(int* pBuffersize, void* pUserData)
{

	RemoteEncoder *remo = NULL;

	if(pUserData)
	{
		remo = (RemoteEncoder * )pUserData;
	}
	else
	{
		cout << " Handle Aquire Bitstream fail!" << endl;
	}
	*pBuffersize = remo->getWidth()*remo->getHeight()*3/2;
	return remo->getCharBuf();		

}

static void __stdcall HandleOnBeginFrame(const NVVE_BeginFrameInfo* frameInfo, void* pUserData)
{
}

static void __stdcall HandleOnEndFrame(const NVVE_EndFrameInfo* frameInfo, void* pUserData)
{
	//HandleReleaseBitStream();
	RemoteEncoder *remo;

	if(pUserData)
	{
		remo = (RemoteEncoder * )pUserData;
	}
}

static void __stdcall HandleReleaseBitStream(int nBytesInBuffer, unsigned char* cb, void* pUserData)
{
	RemoteEncoder *remo = NULL;
	if(pUserData)
	{
		remo = (RemoteEncoder * )pUserData;
	}
	else
	{
		cout << " Handle Release Bitstream fail!" << endl;
	}

	if (remo)
	{
		if(remo->getMeasure())
		{
			SYSTEMTIME st;
			GetLocalTime(&st);

			UINT8 id = remo->getPicId();
			char* msg = new char[sizeof(UINT8) + sizeof(unsigned char) * nBytesInBuffer + sizeof(int) + sizeof(DWORD) * 2];
			memcpy(msg, &FRAME_DATA_MEASURE, sizeof(UINT8));
			memcpy(msg + sizeof(UINT8), &nBytesInBuffer, sizeof(int));
			memcpy(msg + sizeof(UINT8) + sizeof(int), cb, sizeof(unsigned char) * nBytesInBuffer);
			memcpy(msg + sizeof(UINT8) + sizeof(int) + nBytesInBuffer * sizeof(unsigned char), &(st.wSecond), sizeof(DWORD));
			memcpy(msg + sizeof(UINT8) + sizeof(int) + nBytesInBuffer * sizeof(unsigned char) + sizeof(DWORD), &id, sizeof(UINT8));

			remo->incPicID();

			int numBytes = remo->getClient()->Send(msg, sizeof(UINT8) + sizeof(unsigned char) * nBytesInBuffer + sizeof(int) + sizeof(DWORD) * 2);
			delete [] msg;
		}
		else
		{
			char* msg = new char[sizeof(UINT8) + sizeof(unsigned char) * nBytesInBuffer + sizeof(int)];
			memcpy(msg, &FRAME_DATA, sizeof(UINT8));
			memcpy(msg + sizeof(UINT8), &nBytesInBuffer, sizeof(int));
			memcpy(msg + sizeof(UINT8) + sizeof(int), cb, sizeof(unsigned char) * nBytesInBuffer);

			int numBytes = remo->getClient()->Send(msg, sizeof(UINT8) + sizeof(unsigned char) * nBytesInBuffer + sizeof(int));
			delete [] msg;
		}
	}
}



bool RemoteEncoder::setCBFunctions(NVVE_CallbackParams *pCB, void *pUserData)
{
	m_NVCB = *pCB;
	// 	cout << "kommen wir igentwann hier rein?<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< " << endl;
	//	cin.get();
	//Register the callback structure functions
	NVRegisterCB(m_CudaEncoder, m_NVCB, this); //register the callback structure

	return true;
}






/************************************************************************/
/* Remote Encoder                                                       */
/************************************************************************/


RemoteEncoder::RemoteEncoder(int o_width, int o_height)
{
	//m_CudaEncoder = this;
	m_EncoderParams = new NVEncoderParams;
	latencyMeasure = false;
	picId = 0;

	//outbuffer usw. nur um encode in eine datei zu schreiben
	outBuf = new unsigned char[o_width*o_height*3/2]; //passt nocht nicht muss * 3/2 oder so sein
	ZeroMemory(outBuf, o_width*o_height*3/2*sizeof(unsigned char));

	memset(m_EncoderParams, 0, sizeof(NVEncoderParams));

	/************************************************************************/
	/* Setting up Encodingparams Struct                                     */
	/************************************************************************/

	// 4 equals H264 Codec
	m_EncoderParams->iCodecType = NV_CODEC_TYPE_H264;

	// Frameproperties
	m_EncoderParams->iOutputSize[0] = o_width;
	m_EncoderParams->iOutputSize[1] = o_height;

	m_EncoderParams->iInputSize[0] = o_width;
	m_EncoderParams->iInputSize[1] = o_height;

	width = o_width;
	height = o_height;

	m_EncoderParams->iAspectRatio[0] = ASPECT_WIDTH;
	m_EncoderParams->iAspectRatio[1] = ASPECT_HEIGHT;

	// Fragwürdig:
	m_EncoderParams->iP_Interval = 1;
	m_EncoderParams->iIDR_Period = 1;
	m_EncoderParams->iDynamicGOP = 1;
	m_EncoderParams->RCType =  RC_VBR;
	m_EncoderParams->iQP_Level_InterB = 31;
	m_EncoderParams->iQP_Level_InterP = 28;
	m_EncoderParams->iQP_Level_Intra = 25;
	m_EncoderParams->iDeblockMode = 1;
	m_EncoderParams->iProfileLevel = 0xff4d;
	m_EncoderParams->DIMode = DI_MEDIAN;
	m_EncoderParams->Presets = ENC_PRESET_AVCHD;
	m_EncoderParams->iDeviceMemInput = 1;


	//Bitrates
	m_EncoderParams->iAvgBitrate =  5 * 100000;
	m_EncoderParams->iPeakBitrate = 5 * 250000;

	//Hardwarestuff
	m_EncoderParams->GPUOffloadLevel = NVVE_GPU_OFFLOAD_ALL;
	m_EncoderParams->MaxOffloadLevel = NVVE_GPU_OFFLOAD_ALL;
	m_EncoderParams->iUseDeviceMem = 1;

	//Frames
	m_EncoderParams->iFrameRate[0] = 30000;
	m_EncoderParams->iFrameRate[1] = 30000;
	m_EncoderParams->iSurfaceFormat = 2;
	m_EncoderParams->iPictureType = 3;
	m_EncoderParams->iDisableCabac = FALSE;

	/************************************************************************/
	/*                                                                      */
	/************************************************************************/


	//Create Encoder
	errorHandling = NVCreateEncoder(&m_CudaEncoder);
	handleHR(errorHandling, "Create Encoder:");


	//Set up Codec
	errorHandling = NVSetCodec(m_CudaEncoder, m_EncoderParams->iCodecType);
	handleHR(errorHandling, "Set Codec:");

	//default
	errorHandling = NVSetDefaultParam(m_CudaEncoder);
	handleHR(errorHandling,"set default params: ");


	//Setting/Getting Params
	BOOL b = FALSE;
	errorHandling = NVGetParamValue(m_CudaEncoder, NVVE_DISABLE_CABAC, &(b));

	errorHandling = NVGetParamValue(m_CudaEncoder, NVVE_GET_GPU_COUNT,  &(m_EncoderParams->GPU_count));
	handleHR(errorHandling, "Get GPU Count:");

	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_FORCE_GPU_SELECTION, &(m_EncoderParams->iForcedGPU));
	handleHR(errorHandling, "Set active GPU:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_GPU_OFFLOAD_LEVEL, &(m_EncoderParams->GPUOffloadLevel));
	handleHR(errorHandling, "Set offload Level:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_OUT_SIZE, &(m_EncoderParams->iOutputSize));
	handleHR(errorHandling, "Setting Outsize:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_IN_SIZE, &(m_EncoderParams->iInputSize));
	handleHR(errorHandling, "Setting Input:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_ASPECT_RATIO, &(m_EncoderParams->iAspectRatio));
	handleHR(errorHandling, "Setting Aspect Ratio:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_FIELD_ENC_MODE, &(m_EncoderParams->Fieldmode));
	handleHR(errorHandling, "Setting Fieldmode:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_P_INTERVAL, &(m_EncoderParams->iP_Interval));
	handleHR(errorHandling, "Setting P Interval:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_IDR_PERIOD, &(m_EncoderParams->iIDR_Period));
	handleHR(errorHandling, "Setting IDR:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_DYNAMIC_GOP, &(m_EncoderParams->iDynamicGOP));
	handleHR(errorHandling, "Setting Dynamic GOP:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_RC_TYPE, &(m_EncoderParams->RCType));
	handleHR(errorHandling, "Setting RCType:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_AVG_BITRATE, &(m_EncoderParams->iAvgBitrate));
	handleHR(errorHandling, "Setting Avg Bitrate:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_PEAK_BITRATE, &(m_EncoderParams->iPeakBitrate));
	handleHR(errorHandling, "Setting Peak Bitrate:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_QP_LEVEL_INTRA, &(m_EncoderParams->iQP_Level_Intra));
	handleHR(errorHandling, "Setting QP Intra:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_QP_LEVEL_INTER_B, &(m_EncoderParams->iQP_Level_InterB));
	handleHR(errorHandling, "Setting QP Inter B:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_QP_LEVEL_INTER_P, &(m_EncoderParams->iQP_Level_InterP));
	handleHR(errorHandling, "Setting QP Inter P:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_FRAME_RATE, &(m_EncoderParams->iFrameRate));
	handleHR(errorHandling, "Setting FrameRate:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_DEBLOCK_MODE, &(m_EncoderParams->iDeblockMode));
	handleHR(errorHandling, "Setting DeblockMode:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_PROFILE_LEVEL, &(m_EncoderParams->iProfileLevel));
	handleHR(errorHandling, "Setting ProfileLevel:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_FORCE_INTRA, &(m_EncoderParams->iForceIntra));
	handleHR(errorHandling, "Setting ForceIntra:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_FORCE_IDR, &(m_EncoderParams->iForceIDR));
	handleHR(errorHandling, "Setting ForceIDR:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_CLEAR_STAT, &(m_EncoderParams->iClearStat));
	handleHR(errorHandling, "Setting ClearStat:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_SET_DEINTERLACE, &(m_EncoderParams->DIMode));
	handleHR(errorHandling, "Setting DI:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_PRESETS, &(m_EncoderParams->Presets));
	handleHR(errorHandling, "Setting Presets:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_DISABLE_CABAC, &(m_EncoderParams->iDisableCabac));
	handleHR(errorHandling, "Setting CABAC!1:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_CONFIGURE_NALU_FRAMING_TYPE, &(m_EncoderParams->iNaluFramingType));
	handleHR(errorHandling, "Setting nalu framing Type:");
	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_DISABLE_SPS_PPS, &(m_EncoderParams->iDisableSPSPPS));
	handleHR(errorHandling, "Setting SPSPPS:");


	/************************************************************************/
	/* Create Cuda Context                                                  */
	/************************************************************************/
	createCuda();

	/************************************************************************/
	/*                                                                      */
	/************************************************************************/
	memset(&m_cbParams, 0, sizeof(NVVE_CallbackParams));
	m_cbParams.pfnacquirebitstream = HandleAquireBitStream;
	m_cbParams.pfnonbeginframe = HandleOnBeginFrame;
	m_cbParams.pfnonendframe = HandleOnEndFrame;
	m_cbParams.pfnreleasebitstream = HandleReleaseBitStream;

	setCBFunctions(&m_cbParams, (void*) m_CudaEncoder);

	errorHandling = NVCreateHWEncoder(m_CudaEncoder);
	handleHR(errorHandling, "Creating HWEncoder:");

	/************************************************************************/
	/* Setting Up the Encoder                                               */
	/************************************************************************/
	setEncoderParams();

	/************************************************************************/
	/*                                                                      */
	/************************************************************************/


	//NVRegisterCB(m_CudaEncoder, m_cbParams, m_userData);
}


RemoteEncoder::~RemoteEncoder(void)
{

}

void RemoteEncoder::setEncoderParams()
{
	ZeroMemory(&m_efParams, sizeof(NVVE_EncodeFrameParams));
	m_efParams.Height = m_EncoderParams->iOutputSize[1];
	m_efParams.Width = m_EncoderParams->iOutputSize[0];
	m_efParams.Pitch = m_EncoderParams->iOutputSize[0];//(m_EncoderParams->nDeviceMemPitch ? m_EncoderParams->nDeviceMemPitch : m_EncoderParams->iOutputSize[0]);
	m_efParams.PictureStruc = (NVVE_PicStruct) m_EncoderParams->iPictureType;
	m_efParams.SurfFmt = YV12;//(NVVE_SurfaceFormat) m_EncoderParams->iSurfaceFormat;
	m_efParams.progressiveFrame = (m_EncoderParams->iSurfaceFormat == 3) ? 1 : 0;
	m_efParams.repeatFirstField = 0;
	m_efParams.topfieldfirst = (m_EncoderParams->iSurfaceFormat == 1) ? 1 : 0;
	m_efParams.picBuf = NULL;
	m_efParams.bLast = 0;
}

bool RemoteEncoder::encodePB()
{
	errorHandling = NVEncodeFrame(m_CudaEncoder, &m_efParams, 0, NULL);
	if(errorHandling != S_OK)
	{
		handleHR(errorHandling, "FrameEncoding:");
		return false;
	}

	return true;
}


void RemoteEncoder::handleHR(HRESULT hr, const char* c)
{
	cout << c << endl;
	switch (hr){
	case S_OK:
		cout << "S_OK!" << endl;
		break;
	case E_NOTIMPL:
		cout << "E_NOTIMPL" << endl;
		break;
	case E_UNEXPECTED:
		cout << "E_UNEXPECTED" << endl;
		break;
	case E_POINTER:
		cout << "E_POINTER" << endl;
		break;
	case E_FAIL:
		cout << "E_FAIL" << endl;
		break;
	case E_NOINTERFACE:
		cout << "E_NOINTERFACE" << endl;
		break;
	case E_OUTOFMEMORY:
		cout << "E_OUTOFMEMORY" << endl;
		break;
	default:
		cout << "Unknown Error -> Const Nr: " << hr << endl;
		break;
	}
	cout << endl;
	errorHandling = S_OK;
}

void RemoteEncoder::createCuda()
{
	//Creating Cuda Basics
	cout << "\n\nTrying to initialize Cuda\n" << endl;
	CUresult cuRes = cuInit(0);
	handleCudaError(cuRes, "Init Cuda:");
	cuRes = cuDeviceGet(&m_cuDevice, m_EncoderParams->iForcedGPU);
	handleCudaError(cuRes, "Creating Cuda Device:");
	cuRes = cuCtxCreate(&m_cuContext, CU_CTX_BLOCKING_SYNC, m_cuDevice);
	handleCudaError(cuRes, "Creating Cuda Context:");

	//Setup Frameprops & Memory...
	unsigned int widthInBytes = m_EncoderParams->iInputSize[0];
	unsigned int height = (unsigned int)(m_EncoderParams->iInputSize[1] * 12) >> 3;
	size_t cuPitch;
	cuRes = cuMemAllocPitch(&dptr, &cuPitch, widthInBytes, height, 16);
	handleCudaError(cuRes, "MemAllocPitch:");

	m_EncoderParams->nDeviceMemPitch = cuPitch;

	cuRes = cuvidCtxLockCreate(&m_cuCtxLock, m_cuContext);
	handleCudaError(cuRes, "Create  CudaVideoLock:");

	errorHandling = NVSetParamValue(m_CudaEncoder, NVVE_DEVICE_MEMORY_INPUT, &(m_EncoderParams->iUseDeviceMem));
	handleHR(cuRes, "Forced to use chosen Devicemem:");
}

void RemoteEncoder::handleCudaError(CUresult cuRes, const char* c)
{
	cout << c << endl;
	switch(cuRes)
	{
	case CUDA_SUCCESS:
		cout << "CUDA_SUCCESS" << endl;
		break;
	case CUDA_ERROR_INVALID_VALUE:
		cout << "CUDA_ERROR_INVALID_VALUE" << endl;
		break;
	case CUDA_ERROR_INVALID_DEVICE:
		cout << "CUDA_ERROR_INVALID_DEVICE" << endl;
		break;
	case CUDA_ERROR_DEINITIALIZED:
		cout << "CUDA_ERROR_DEINITIALIZED" << endl;
		break;
	case CUDA_ERROR_NOT_INITIALIZED:
		cout << "CUDA_ERROR_NOT_INITIALIZED" << endl;
		break;
	case CUDA_ERROR_INVALID_CONTEXT:
		cout << "CUDA_ERROR_INVALID_CONTEXT" << endl;
		break;
	case CUDA_ERROR_OUT_OF_MEMORY:
		cout << "CUDA_ERROR_OUT_OF_MEMORY" << endl;
		break;
	case CUDA_ERROR_UNKNOWN:
		cout << "CUDA_ERROR_UNKNOWN" << endl;
		break;
	default:
		cout << "ERROR_UNCATCHED -- NO SPECIAL CASE AVAILABLE" << endl;
		break;
	}
	cout << endl;
}

void RemoteEncoder::setDevicePtr(CUdeviceptr dptr)
{
	this->dptr = dptr;
}
