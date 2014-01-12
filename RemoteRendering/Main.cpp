#include "Main.h"
#include "device_launch_parameters.h"



void initCuda()
{
	//Creating Cuda Basics
	cout << "\n\nTrying to initialize Cuda\n" << endl;
	CUresult cuRes = cuInit(0);
	remo->handleCudaError(cuRes, "Init Cuda:");
	cuRes = cuDeviceGet(&cuDev, 0);
	remo->handleCudaError(cuRes, "Creating Cuda Device:");	
	cuRes = cuCtxCreate(&cuCtx, CU_CTX_BLOCKING_SYNC, cuDev);	
	remo->handleCudaError(cuRes, "Creating Cuda Context:");

	memset( &prop, 0, sizeof(cudaDeviceProp));
	prop.major = 1;
	prop.minor = 0;
	cudaChooseDevice(&cuDev, &prop);
	cudaGLSetGLDevice(cuDev);


	//Speziell für PixelBuffer
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo); //könnte auch arraybuffer sein
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_DRAW);//GL_STREAM_READ);
	/*glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);*/

	// Cuda Device setzen
	cudaError_t error = cudaGraphicsGLRegisterBuffer(&resource, pbo, cudaGraphicsRegisterFlagsReadOnly);
	if(error == cudaSuccess)
	{
		std::cout << "Registering Cuda Resource: OK" << std::endl;
	}
	CUDA_SAFE_CALLING(cudaMalloc((void**)&d_yuv, arraySize*sizeof(unsigned char)));
}

GLuint createTexture(const char* fileName)
{
	GLuint texID;
	int width, height, imgFormat, internalFormat;
	float* imgData = getImage(fileName, &height, &width, &imgFormat);
	switch (imgFormat)
	{
	case GL_RED: internalFormat = GL_R8; break;
	case GL_RG: internalFormat = GL_RG8; break;
	case GL_RGB: internalFormat = GL_RGB8; break;
	case GL_RGBA: internalFormat = GL_RGBA8; break;
	default: fprintf(stderr, "\n Cannot get ImgType \n"); break;
	}
	glGenTextures(1, &texID);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texID);
	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, imgFormat, GL_FLOAT, (void*) imgData);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glGenerateMipmap(GL_TEXTURE_2D);
	glUniform1i(glGetUniformLocation(programID, "colorTex"), 0);
	earth->draw();
	return texID;
}

inline void processKeyOps()
{
	printf("MoveDir: x %f y %f z %f \n", moveDir.x, moveDir.y, moveDir.z);
	if(keyStates['w'] || keyStates['W'])
	{
		cam->move(1 * MOVESPEED, 0, 0);
	}
	else moveDir.z = 0;
	if(keyStates['s'] || keyStates['S'])
	{
		cam->move(-1 * MOVESPEED, 0, 0);
	}
	else moveDir.z = 0;	
	if(keyStates['a'] || keyStates['A'])
	{
		cam->move(0, 1 * MOVESPEED, 0);
	}
	else moveDir.x = 0;
	if(keyStates['d'] || keyStates['D'])
	{
		cam->move(0, -1 * MOVESPEED, 0);
	}
	else moveDir.x = 0;
	if(keyStates[' '])
	{
		cam->move(0, 0, 1 * MOVESPEED);
	}
	else moveDir.y = 0;
	if(keyStates['c'] || keyStates['C'])
	{
		cam->move(0, 0, -1 * MOVESPEED);
	}
	else moveDir.y = 0;
	if (keySpecialStates[GLUT_KEY_UP])
	{
	}
	if (keySpecialStates[GLUT_KEY_DOWN])
	{
	}
	if (keySpecialStates[GLUT_KEY_LEFT])
	{
	}
	if (keySpecialStates[GLUT_KEY_RIGHT])
	{
	}
	if (keySpecialStates[GLUT_KEY_END])
	{
	}
	if(keySpecialStates[GLUT_KEY_F12])
	{
		m_continue = false;
		cout << "Shutting down" << endl;

		char* message = new char(sizeof(char));
		memcpy(message, &SHUTDOWN_CONNECTION, sizeof(UINT8));
		int error = serverSocket->Send(message, sizeof(UINT8));
		if(error  <= 1)
			cout << "Fehler beim Versenden der Shutdown Nachricht" << endl;
	}
}

void drawScene(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	GetSystemTime(&st);
	currentTimeMS = st.wMilliseconds; 
	WORD timeDif;
	if((timeDif = currentTimeMS - lastTimeMS) < 55)
	{
		Sleep(55 - timeDif);
	}

	//GLSHIT here
	glUseProgram(programID);
	glUniformMatrix4fv(modelLocation, 1, false, glm::value_ptr(cam->getView()));
	cam->move(0.04 * moveDir.z, 0.04 * moveDir.x, 0.04 * moveDir.y);
	glm::mat4 viewProj = cam->getProjection() * cam->getView();
	glUniformMatrix4fv(viewProjLoc, 1, false, glm::value_ptr(viewProj));
	earth->draw();
	glutSwapBuffers();

	glFinish();
	//Buffer bei Cuda anmelden

	glReadBuffer(GL_BACK);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	cudaError_t r = cudaSuccess;
	

	CUDA_SAFE_CALLING(cudaGraphicsMapResources(1, &resource, NULL));
	CUDA_SAFE_CALLING(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, NULL, resource));
	callKernel(width,height,d_yuv, devPtr);
	CUDA_SAFE_CALLING(cudaDeviceSynchronize());
	CUDA_SAFE_CALLING(cudaGraphicsUnmapResources(1, &resource, NULL));
	CUDA_SAFE_CALLING(cudaMemcpy( &yuv[0], d_yuv,  yuv.size(), cudaMemcpyDeviceToHost));

	remo->setPicBuf(&yuv[0]);
	remo->encodePB();


	glutPostRedisplay();
	glFinish();

	lastTimeMS = st.wMilliseconds;
}

void timer(int v)
{
	glutPostRedisplay();
	glutTimerFunc(16, timer, v);
}

void initOpenGL(int argc, char** argv)
{
	modelMatrix = glm::mat4x4();
	moveDir = glm::vec3(0.0f, 0.0f, 0.0f);
	bContinue = true;
	cam = new Camera();
	culling = true;
	wireframe = true;
	n = 2;

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(800, 600);
	glutCreateWindow("Decoder!");
	glewInit();

	glutDisplayFunc(drawScene);
	glutTimerFunc(16, timer, 0);

	glViewport(0, 0, 800, 600);
	glPrimitiveRestartIndex(PRIMITIVE_RESTART);
	glEnable(GL_PRIMITIVE_RESTART);
}

int main(int argc, char** argv)
{
	serverSocket = new UdpSocket();
	serverSocket->Create();
	serverSocket->Bind(DEFAULT_IP, DEFAULT_PORT+1);

	cout << "Warte auf eingehende Verbindungen!" << endl;

	char message[DEFAULT_BUFLEN];
	serverSocket->Receive(message, DEFAULT_BUFLEN);
	UINT8 identifier;
	memcpy(&identifier, message, sizeof(UINT8));

	cout << "Identifyer is: " << identifier << endl;

	if(identifier == WINDOW_SIZE) 
	{
		memcpy(&width, message + sizeof(UINT8), sizeof(int));
		memcpy(&height, message + sizeof(UINT8) + sizeof(int), sizeof(int));
	}
	serverSocket->SetToNonBlock();


	initOpenGL(argc, argv);
	remo = new RemoteEncoder(width, height);
	remo->setClientUdp(serverSocket);

	programID = createShaderProgram("shader/Main_VS.glsl", "shader/VertexColor_FS.glsl");
	modelLocation = glGetUniformLocation(programID, "model");
	viewProjLoc = glGetUniformLocation(programID, "viewProj");
	cam->move(-5.0f, 0.0f, 0.0f);
	earth = createSphere(1, 64, 32);
	earthTex = createTexture("textures/earth.jpg");

	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glClearColor(0.3f, 0, 0, 1.0f);




	//init cuda datenfelder
	devPtr = NULL;
	arraySize = width * height* 1.5;
	yuv = vector<unsigned char>(arraySize); 
	d_yuv = new unsigned char[arraySize];

	memset(&yuv[0], 0.0, yuv.size());

	initCuda();

	while(m_continue)
	{
		processKeyOps();
		serverSocket->Receive(message, DEFAULT_BUFLEN);
		int key;
		memcpy(&identifier, message, sizeof(UINT8));
		switch(identifier)
		{
		case KEY_PRESSED:
			memcpy(&key, message+sizeof(UINT8), sizeof(int));
			cout << "KEY HIT: " << key << endl;
			if(key <= 256)
				keyStates[key] = true;
			break;
		case KEY_RELEASED:
			memcpy(&key, message+sizeof(UINT8), sizeof(int));
			cout << "KEY RELASED!" << key << endl;
			if(key <= 256)
				keyStates[key] = false;
			break;
		case SPECIAL_KEY_PRESSED:
			memcpy(&key, message+sizeof(UINT8), sizeof(int));
			cout << "SPECIAL KEY Pressed!" << key << endl;
			if(key <= 246)
				keySpecialStates[key] = true;
			break;
		case SPECIAL_KEY_RELEASED:
			memcpy(&key, message+sizeof(UINT8), sizeof(int));
			cout << "SPECIAL KEY RELASED!" << key << endl;
			if(key <= 246)
				keySpecialStates[key] = false;
			break;
		default:
			break;
		}

		identifier = 0;
		memset(message, 0, DEFAULT_BUFLEN);

		glutMainLoopEvent();
	}
	return 0;
}
