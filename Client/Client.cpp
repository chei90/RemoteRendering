#include "Client.h"


void keyPressed(unsigned char key, int x, int y)
{
	keyStates[key] = true;
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
}

void initCallbacks()
{
	glutKeyboardFunc(keyPressed);
	glutKeyboardUpFunc(keyReleased);
	glutSpecialFunc(specialKeyPressed);
	glutSpecialUpFunc(specialKeyReleased);
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
	glutCreateWindow("Decoder!");
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
		
		//////////////// Test Stuff
		//std::vector<unsigned char> nv = std::vector<unsigned char>(800 * 600 * 3 / 2);
		//std::vector<unsigned char> rgba = std::vector<unsigned char>(800 * 600 * 4);
		//cudaMemcpy(&nv[0], (void*)decodedFrame[active], 800 * 600 * 3 / 2, cudaMemcpyDeviceToHost);
		//gpuErrchk(cudaGetLastError());
		cudaError_t res, error;
		res = cudaGraphicsMapResources(1, &cudaTex);
		{
			//res = cudaMemset(globalMem, 60, 800 * 600 * 4);
			//gpuErrchk(cudaGetLastError());
			cudaArray_t cudaTexData;
			error = cudaGraphicsSubResourceGetMappedArray(&cudaTexData, cudaTex, 0, 0);
			//gpuErrchk(cudaGetLastError());
			callDecode(m_width, m_height, (unsigned char*)decodedFrame[active], globalMem, decodedPitch);
			cudaMemcpyToArray(cudaTexData, 0, 0, globalMem, m_width * m_height * sizeof(unsigned char) * 4, cudaMemcpyDeviceToDevice);
			//gpuErrchk(cudaGetLastError());
			//cudaMemcpy(&rgba[0], globalMem, 800 * 600 * 4, cudaMemcpyDeviceToHost);

			cudaFree(cudaTexData);
		}
		cudaGraphicsUnmapResources(1, &cudaTex);

		m_decoder->unmapFrame(decodedFrame[active]);

		m_queue->releaseFrame(&frame);

	}



	checkCudaErrors(cuCtxPopCurrent(NULL));
}

int main(int argc, char** argv)
{
	cuInit(0);
	initGL(argc, argv);
	//	SOCKET STUFF
	RenderSocket* server = new RenderSocket();
	server->Create();

	std::auto_ptr<FrameQueue> tmp_queue(new FrameQueue);
	m_queue = tmp_queue.release();

	cout << (void*) m_queue << " Client" << endl;

	m_decoder = new Decoder(m_ctx, m_lock, m_queue, m_width, m_height);
	m_decoder->initParser();


	cout << "Client sucht Verbindung..." << endl;
	server->Connect(DEFAULT_IP, DEFAULT_PORT);
	cout << "Client mit Server verbunden!" << endl;


	//PROCESS USER INPUT
	char* message = new char(64);
	char* msgStart = message;
	char* serverMessage = new char[100000];
	CUVIDPARSERDISPINFO f;

	//Sending Window Size
	memcpy(message, &WINDOW_SIZE, sizeof(UINT8));
	memcpy(message + sizeof(UINT8), &m_width, sizeof(int));
	memcpy(message + sizeof(UINT8) + sizeof(int), &m_height, sizeof(int));
	int j = server->Send(message, sizeof(UINT8) + sizeof(int) * 2);
	memset(message, 0, sizeof(UINT8) + sizeof(int) * 2);
	cout << j << " signs sent" << endl;

	while (m_continue)
	{
		memset(serverMessage, 0, 100000);
		msgStart = message;

		server->Receive(serverMessage, 100000);
		UINT8 identifyer;
		memcpy(&identifyer, serverMessage, sizeof(UINT8));
		switch (identifyer)
		{
		case SHUTDOWN_CONNECTION:
			m_continue = false;
			break;
		case FRAME_DATA:
			int size;
			memcpy(&size, serverMessage+sizeof(UINT8), sizeof(int));
			m_decoder->parseData((const unsigned char*)(serverMessage + sizeof(UINT8) + sizeof(int)), size);
			if(m_queue->dequeue(&f))
			{
				copyFrameToTexture(f);
			}
			break;
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

		glutMainLoopEvent();
		memset(message, 0, 64); 
	
	}

	server->Close();

	std::cin.get();

}


