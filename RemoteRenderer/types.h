#pragma once

#include "NVEncodeDataTypes.h"

struct NVEncoderParams
{
	char configFile[256];
	char inputFile[256];
	char outputFile[256];
	int measure_psnr;
	int measure_fps;
	int force_device;
	int iSurfaceFormat;
	int iPictureType;
	int nDeviceMemPitch;

	int                     iCodecType;       //    NVVE_CODEC_TYPE,
	int                     GPU_count;        //    Choose the specific GPU count
	int                     GPU_devID;        //    Choose the specific GPU device ID
	int                     iUseDeviceMem;    //    CUDA with DEVICE_MEMORY_INPUT (for encoding)
	int                     iForcedGPU;       //    NVVE_FORCE_GPU_SELECTION            //F22
	int                     iOutputSize[2];   //    NVVE_OUT_SIZE,
	int                     iInputSize[2];    //    NVVE_IN_SIZE,
	float                   fAspectRatio;     //
	int                     iAspectRatio[3];  //    NVVE_ASPECT_RATIO,
	NVVE_FIELD_MODE         Fieldmode;        //    NVVE_FIELD_ENC_MODE,
	int                     iP_Interval;      //    NVVE_P_INTERVAL,
	int                     iIDR_Period;      //    NVVE_IDR_PERIOD,
	int                     iDynamicGOP;      //    NVVE_DYNAMIC_GOP,
	NVVE_RateCtrlType       RCType;           //    NVVE_RC_TYPE,
	int                     iAvgBitrate;      //    NVVE_AVG_BITRATE,
	int                     iPeakBitrate;     //    NVVE_PEAK_BITRATE,
	int                     iQP_Level_Intra;  //    NVVE_QP_LEVEL_INTRA,
	int                     iQP_Level_InterP; //    NVVE_QP_LEVEL_INTER_P,
	int                     iQP_Level_InterB; //    NVVE_QP_LEVEL_INTER_B,
	int                     iFrameRate[2];    //    NVVE_FRAME_RATE,
	int                     iDeblockMode;     //    NVVE_DEBLOCK_MODE,
	int                     iProfileLevel;    //    NVVE_PROFILE_LEVEL,
	int                     iForceIntra;      //    NVVE_FORCE_INTRA,
	int                     iForceIDR;        //    NVVE_FORCE_IDR,
	int                     iClearStat;       //    NVVE_CLEAR_STAT,
	NVVE_DI_MODE            DIMode;           //    NVVE_SET_DEINTERLACE,
	NVVE_PRESETS_TARGET     Presets;          //    NVVE_PRESETS,
	int                     iDisableCabac;    //    NVVE_DISABLE_CABAC,
	int                     iNaluFramingType; //    NVVE_CONFIGURE_NALU_FRAMING_TYPE
	int                     iDisableSPSPPS;   //    NVVE_DISABLE_SPS_PPS
	NVVE_GPUOffloadLevel    GPUOffloadLevel;  //    NVVE_GPU_OFFLOAD_LEVEL
	NVVE_GPUOffloadLevel    MaxOffloadLevel;  //    NVVE_GPU_OFFLOAD_LEVEL_MAX
	int                     iSliceCnt;        //    NVVE_SLICE_COUNT                    //F19
	int                     iMultiGPU;        //    NVVE_MULTI_GPU                      //F21
	int                     iDeviceMemInput;  //    NVVE_DEVICE_MEMORY_INPUT            //F23

	//    NVVE_STAT_NUM_CODED_FRAMES,
	//    NVVE_STAT_NUM_RECEIVED_FRAMES,
	//    NVVE_STAT_BITRATE,
	//    NVVE_STAT_NUM_BITS_GENERATED,
	//    NVVE_GET_PTS_DIFF_TIME,
	//    NVVE_GET_PTS_BASE_TIME,
	//    NVVE_GET_PTS_CODED_TIME,
	//    NVVE_GET_PTS_RECEIVED_TIME,
	//    NVVE_STAT_ELAPSED_TIME,
	//    NVVE_STAT_QBUF_FULLNESS,
	//    NVVE_STAT_PERF_FPS,
	//    NVVE_STAT_PERF_AVG_TIME,
};

typedef struct
{
	char *name;
	int  params;
} _sNVVEEncodeParams;

static _sNVVEEncodeParams sNVVE_EncodeParams[] =
{
	{ "UNDEFINED", 1 },
	{ "NVVE_OUT_SIZE", 2 },
	{ "NVVE_ASPECT_RATIO", 3 },
	{ "NVVE_FIELD_ENC_MODE", 1 },
	{ "NVVE_P_INTERVAL", 1 },
	{ "NVVE_IDR_PERIOD", 1 },
	{ "NVVE_DYNAMIC_GOP", 1 },
	{ "NVVE_RC_TYPE", 1 },
	{ "NVVE_AVG_BITRATE", 1 },
	{ "NVVE_PEAK_BITRATE", 1 },
	{ "NVVE_QP_LEVEL_INTRA", 1 },
	{ "NVVE_QP_LEVEL_INTER_P", 1 },
	{ "NVVE_QP_LEVEL_INTER_B", 1 },
	{ "NVVE_FRAME_RATE", 2 },
	{ "NVVE_DEBLOCK_MODE", 1 },
	{ "NVVE_PROFILE_LEVEL", 1 },
	{ "NVVE_FORCE_INTRA (DS)", 1 },            //DShow only
	{ "NVVE_FORCE_IDR   (DS)", 1 },            //DShow only
	{ "NVVE_CLEAR_STAT  (DS)", 1 },            //DShow only
	{ "NVVE_SET_DEINTERLACE", 1 },
	{ "NVVE_PRESETS", 1 },
	{ "NVVE_IN_SIZE", 2 },
	{ "NVVE_STAT_NUM_CODED_FRAMES (DS)", 1 },       //DShow only
	{ "NVVE_STAT_NUM_RECEIVED_FRAMES (DS)", 1 },    //DShow only
	{ "NVVE_STAT_BITRATE (DS)", 1 },                //DShow only
	{ "NVVE_STAT_NUM_BITS_GENERATED (DS)", 1 },     //DShow only
	{ "NVVE_GET_PTS_DIFF_TIME (DS)", 1 },           //DShow only
	{ "NVVE_GET_PTS_BASE_TIME (DS)", 1 },           //DShow only
	{ "NVVE_GET_PTS_CODED_TIME (DS)", 1 },          //DShow only
	{ "NVVE_GET_PTS_RECEIVED_TIME (DS)", 1 },       //DShow only
	{ "NVVE_STAT_ELAPSED_TIME (DS)", 1 },           //DShow only
	{ "NVVE_STAT_QBUF_FULLNESS (DS)", 1 },          //DShow only
	{ "NVVE_STAT_PERF_FPS (DS)", 1 },               //DShow only
	{ "NVVE_STAT_PERF_AVG_TIME (DS)", 1 },          //DShow only
	{ "NVVE_DISABLE_CABAC", 1 },
	{ "NVVE_CONFIGURE_NALU_FRAMING_TYPE", 1 },
	{ "NVVE_DISABLE_SPS_PPS", 1 },
	{ "NVVE_SLICE_COUNT", 1 },
	{ "NVVE_GPU_OFFLOAD_LEVEL", 1 },
	{ "NVVE_GPU_OFFLOAD_LEVEL_MAX", 1 },
	{ "NVVE_MULTI_GPU", 1 },
	{ "NVVE_GET_GPU_COUNT", 1 },
	{ "NVVE_GET_GPU_ATTRIBUTES", 1 },
	{ "NVVE_FORCE_GPU_SELECTION", 1 },
	{ "NVVE_DEVICE_MEMORY_INPUT", 1 },
	{ "NVVE_DEVICE_CTX_LOCK", 1 }
};
