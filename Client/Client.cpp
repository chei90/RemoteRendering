#include "Client.h"


void keyPressed(unsigned char key, int x, int y)
{
	keyStates[key] = true;

	measure = true;
	GetLocalTime(&st);
	sec = st.wSecond;
	msec = st.wMilliseconds;

	if(key == '2')
		printLatency ^= true;
}

void keyReleased(unsigned char key, int x, int y)
{
	keyStates[key] = false;
}

void specialKeyPressed(int key, int x, int y)
{
	keySpecialStates[key] = true;
}

void specialKeyReleased(int key, int x, int y)
{
	keySpecialStates[key] = false;
}

void motionFunc(int x, int y)
{
	mouseDx = x - mouseDx;
	mouseDy = y - mouseDy;
}

void mouseFunc(int button, int state, int dx, int dy)
{
	if(button == GLUT_LEFT_BUTTON)
	{
		pressed = !state;
	}
}

void render()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindTexture(GL_TEXTURE_2D, currentFrameTex);
	glBegin(GL_QUADS);
		
		//Unten links
		glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
		//Unten rechts
		glTexCoord2f(1.0f, 1.0f); glVertex2f(1.0f, -1.0f);
		//Oben rechts
		glTexCoord2f(1.0f, 0.0f); glVertex2f(1.0f, 1.0f);
		//Oben links
		glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, 1.0f);
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
	glutSwapBuffers();
	glutPostRedisplay();


	if(measure)
	{
		unsigned char* tmpBuf = new unsigned char[100];
		cudaMemcpy(tmpBuf, globalMem, 100 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

		int cnt;
		for(int i = 0; i < 100; i++)
		{
			if(tmpBuf[i] == 0) cnt++;
		}
		if(cnt > 40)
		{
			GetLocalTime(&st);
			DWORD tmpmsec = st.wMilliseconds;
			if((st.wSecond - sec) > 0)
				tmpmsec += 1000;
			printf("----------------------------------- \n");
			printf("Latency: %d ms \n", tmpmsec - msec);
			printf("----------------------------------- \n");

			measure = false;
		}
	}
}

void initCallbacks()
{
	glutKeyboardFunc(keyPressed);
	glutKeyboardUpFunc(keyReleased);
	glutSpecialFunc(specialKeyPressed);
	glutSpecialUpFunc(specialKeyReleased);
	glutMotionFunc(motionFunc);
	glutMouseFunc(mouseFunc);
	glutDisplayFunc(render);
}

void checkCuErrors(CUresult res, const char* msg)
{
	if(res !=  CUDA_SUCCESS)
		std::cout << "Something went wrong in " << msg << std::endl;
}

void initGL(int argc, char** argv)
{
	int deviceCount = 0;
	char deviceName [256];

	checkCuErrors(cuDeviceGetCount(&deviceCount), "initGL - cuDeviceGetCount");
	for(int i = 0; i < deviceCount; i++)
	{
		checkCuErrors(cuDeviceGet(&m_device, i), "initGL - cuDeviceGet");
		checkCuErrors(cuDeviceGetName(deviceName, 256, m_device), "initGL -cuDeviceGetName");
	}

	cudaGLSetGLDevice(m_device);

	printf("Device %d: %s is used!", m_device, deviceName);


	//----------------------------------------------------------------------------------//


	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(m_width, m_height);
	glutCreateWindow("RemoteRenderingClient!");
	glewInit();	
	initCallbacks();
	
	
	checkCuErrors(cuGLCtxCreate(&m_ctx, CU_CTX_BLOCKING_SYNC, m_device), "initGL - cuGLCtxCreate");
	checkCuErrors(cuvidCtxLockCreate(&m_lock, m_ctx), "initGL - cuvidCtxLockCreate");

	cuCtxPushCurrent(m_ctx);

	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &currentFrameTex);
	glBindTexture(GL_TEXTURE_2D, currentFrameTex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_width, m_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_2D, 0);
	cudaError_t res = cudaGraphicsGLRegisterImage(&cudaTex, currentFrameTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);

	if(res != cudaSuccess)
		cout << "Something went wrong while registering cuda Image" << endl;
	else
		cout << "Resources bound successfully!" << endl;

	res = cudaMalloc((void**)&globalMem, m_width * m_height * sizeof(unsigned char) * 4);

	cuCtxPopCurrent(NULL);


	glViewport(0, 0, m_width, m_height);
	glClearColor(0, 0, 1, 1);
	glClearDepth(1);
}

void copyFrameToTexture(CUVIDPARSERDISPINFO frame)
{
	CCtxAutoLock lck(m_lock);
	CUresult res = cuCtxPushCurrent(m_ctx);
	CUdeviceptr decodedFrame[2] = { 0, 0};
	CUdeviceptr interopFrame[2] = { 0, 0};

	int numFields = (frame.progressive_frame? (1) : (2+frame.repeat_first_field));

	//Fragwürdig
	unsigned int repeats;

	for(int active = 0; active < numFields; active++)
	{
		repeats = frame.repeat_first_field;
		CUVIDPROCPARAMS procParams;
		memset(&procParams, 0, sizeof(CUVIDPROCPARAMS));

		procParams.progressive_frame = frame.progressive_frame;
		procParams.second_field = active;
		procParams.top_field_first = frame.top_field_first;
		procParams.unpaired_field = (numFields == 1);

		unsigned int width = 0, height = 0, decodedPitch = 0;

		m_decoder->mapFrame(frame.picture_index, &decodedFrame[active], &decodedPitch, &procParams);
		cudaError_t res, error;
		res = cudaGraphicsMapResources(1, &cudaTex);
		{
			cudaArray_t cudaTexData;
			error = cudaGraphicsSubResourceGetMappedArray(&cudaTexData, cudaTex, 0, 0);
			callDecode(m_width, m_height, (unsigned char*)decodedFrame[active], globalMem, decodedPitch);
			cudaMemcpyToArray(cudaTexData, 0, 0, globalMem, m_width * m_height * sizeof(unsigned char) * 4, cudaMemcpyDeviceToDevice);
			cudaFree(cudaTexData);
		}
		cudaGraphicsUnmapResources(1, &cudaTex);
		m_decoder->unmapFrame(decodedFrame[active]);
		m_queue->releaseFrame(&frame);

	}

	
	cuCtxPopCurrent(NULL);
}

int main(int argc, char** argv)
{
	ConfigFile cf("clientConfig.ini");

	std::string sIp = cf.Value("server", "ip");
	int sPort = cf.Value("server", "port");

	std::string cIp = cf.Value("client", "ip");
	int cPort = cf.Value("client", "port");

	m_width = cf.Value("resolution", "width");
	m_height = cf.Value("resolution", "height");

	picId = 0;
	picNum = 0;

	cuInit(0);
	initGL(argc, argv);
	//	SOCKET STUFF
	TcpSocket* server = new TcpSocket();
	server->Create();

	std::auto_ptr<FrameQueue> tmp_queue(new FrameQueue);
	m_queue = tmp_queue.release();

	m_decoder = new Decoder(m_ctx, m_lock, m_queue, m_width, m_height);
	m_decoder->initParser();

	server->Bind(cIp, cPort);
	bool b = server->Connect(sIp, sPort);
	printf("Connection succeeded? %d \n", b);
	//PROCESS USER INPUT
	char* message = new char[64];
	char* msgStart = message;
	char* serverMessage = new char[100000];
	CUVIDPARSERDISPINFO f;

	picNum = 0;
	measure = false;

	//Sending Window Size
	memcpy(message, &WINDOW_SIZE, sizeof(UINT8));
	memcpy(message + sizeof(UINT8), &m_width, sizeof(int));
	memcpy(message + sizeof(UINT8) + sizeof(int), &m_height, sizeof(int));
	int j = server->Send(message, sizeof(UINT8) + sizeof(int) * 2);
	int err = WSAGetLastError();
	printf("Error Code: %d", err);
	memset(message, 0, sizeof(UINT8) + sizeof(int) * 2);
	cout << j << " signs sent" << endl;

	SYSTEMTIME fps;
	DWORD fpsSec = 0, fpsMsec = 0;
	GetSystemTime(&fps);
	long fpsCounter = 0;
	long bandWidth = 0;
	bool printLatency = true;

	while (m_continue)
	{
		GetSystemTime(&fps);
		fpsCounter++;
		if(fps.wSecond != fpsSec && printLatency)
		{
			fpsSec++;
			float bandWidthPrint = bandWidth / 1024.0f / 1024.0f;
			printf("Fps: %d  Bandwidth: %f.3\n", fpsCounter, bandWidthPrint);
			fpsCounter = 0;
			bandWidth = 0;
			if(fpsSec == 60)
			{
				fpsSec = 0;
			}
		}

		memset(serverMessage, 0, 100000);
		int i = server->Receive(serverMessage, 100000);
		bandWidth += i;
		message = msgStart;

		UINT8 identifyer;
		memcpy(&identifyer, serverMessage, sizeof(UINT8));
		switch (identifyer)
		{
		case SHUTDOWN_CONNECTION:
			server->SetToNonBlock();
			m_continue = false;
			break;
		case FRAME_DATA:
			{
			int size;
			memcpy(&size, serverMessage+sizeof(UINT8), sizeof(int));
			m_decoder->parseData((const unsigned char*)(serverMessage + sizeof(UINT8) + sizeof(int)), size);
			if(m_queue->dequeue(&f))
			{
				copyFrameToTexture(f);
			}
			break;
			}
		case FRAME_DATA_MEASURE:
			{
			int size;
			memcpy(&size, serverMessage+sizeof(UINT8), sizeof(int));
			m_decoder->parseData((const unsigned char*)(serverMessage + sizeof(UINT8) + sizeof(int)), size);
			if(m_queue->dequeue(&f))
			{
				copyFrameToTexture(f);
			}
			DWORD lSec = 0; DWORD lMsec = 0;
			memcpy(&lSec, serverMessage + sizeof(UINT8) + sizeof(int) + sizeof(unsigned char) * size, sizeof(DWORD));
			memcpy(&remotePicId, serverMessage + sizeof(UINT8) + sizeof(int) + sizeof(unsigned char) * size + sizeof(DWORD), sizeof(UINT8));
			break;
			}
		default:
			break;
		}

		for (int i = 0; i < 256; i++)
		{
			if (keyStates[i] && !tmpKeyStates[i])
			{
				tmpKeyStates[i] = true;
				memcpy(message, &KEY_PRESSED, sizeof(UINT8));
				message++;
				memcpy(message, &i, sizeof(char));
				message++;

				std::cout << msgStart[1] << std::endl;
			}
			if (!keyStates[i] && tmpKeyStates[i])
			{
				tmpKeyStates[i] = false;

				memcpy(message, &KEY_RELEASED, sizeof(UINT8));
				message++;
				memcpy(message, &i, sizeof(char));
				message++;
			}
		}
		for (int i = 0; i < 246; i++)
		{
			if (keySpecialStates[i] && !tmpKeySpecialStates[i])
			{
				tmpKeySpecialStates[i] = true;

				memcpy(message, &SPECIAL_KEY_PRESSED, sizeof(UINT8));
				message++;
				memcpy(message, &i, sizeof(char));
				message++;

				std::cout << msgStart[1] << std::endl;
			}
			if (!keySpecialStates[i] && tmpKeySpecialStates[i])
			{
				tmpKeySpecialStates[i] = false;

				memcpy(message, &SPECIAL_KEY_RELEASED, sizeof(UINT8));
				message++;
				memcpy(message, &i, sizeof(char));
				message++;
			}
		}

		int size = message - msgStart;
		if (size > 0)
		{
			int i = server->Send(msgStart, size);
		}

		message = msgStart;
		memset(message, 0, 64); 

		if(pressed)
		{
			tmpPressed = true;
			memcpy(message, &MOUSE_PRESSED, sizeof(UINT8));
			message++;
			memcpy(message, &mouseDx, sizeof(int));
			message += sizeof(int);
			memcpy(message, &mouseDy, sizeof(int));
			message += sizeof(int);
		}
		if(!pressed && tmpPressed)
		{
			tmpPressed = false;
			memcpy(message, &MOUSE_RELEASED, sizeof(UINT8));
			message++;
		}
		size = message - msgStart;
		if(size > 0)
		{
			server->Send(msgStart, size);
		}

		memset(message, 0, 64);
		glutMainLoopEvent();	
	}

	server->Close();
	delete server;
	delete m_queue;
	delete [] serverMessage;
	delete [] message;
	delete m_decoder;
}


