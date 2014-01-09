#include "Main.h"
#include "device_launch_parameters.h"



vector<double> calculateNormals(vector<double>& vertices, vector<int>& indices)
{
	int size = vertices.size();
	int iSize = indices.size();

	float* normals = new float[size];
	for (int i = 0; i < size; i++)
		normals[i] = 0;

	const int x = 0, y = 1, z = 2;

	for (int i = 0; i < iSize; i += 3)
	{
		double v1[3];
		double v2[3];
		double p0[3];
		double p1[3];
		double p2[3];
		double normal[3];

		p0[0] = vertices.at(3 * indices.at(i) + x);
		p0[1] = vertices.at(3 * indices.at(i) + y);
		p0[2] = vertices.at(3 * indices.at(i) + z);

		p1[0] = vertices.at(3 * indices.at(i + 1) + x);
		p1[1] = vertices.at(3 * indices.at(i + 1) + y);
		p1[2] = vertices.at(3 * indices.at(i + 1) + z);

		p2[0] = vertices.at(3 * indices.at(i + 2) + x);
		p2[1] = vertices.at(3 * indices.at(i + 2) + y);
		p2[2] = vertices.at(3 * indices.at(i + 2) + z);

		v1[0] = p1[0] - p0[0];
		v1[1] = p1[1] - p0[1];
		v1[2] = p1[2] - p0[2];

		v2[0] = p2[0] - p1[0];
		v2[1] = p2[1] - p1[1];
		v2[2] = p2[2] - p1[2];

		normal[0] = v1[1] * v2[2] - v1[2] * v2[1];
		normal[1] = v1[2] * v2[0] - v1[0] * v2[2];
		normal[2] = v1[0] * v2[1] - v1[1] * v2[0];

		for (int j = 0; j < 3; j++)
		{
			normals[3 * indices.at(i + j) + x] = normals[3 * indices.at(i + j) + x] + normal[0];
			normals[3 * indices.at(i + j) + y] = normals[3 * indices.at(i + j) + y] + normal[1];
			normals[3 * indices.at(i + j) + z] = normals[3 * indices.at(i + j) + z] + normal[2];
		}
	}

	vector<double> ns;
	for (int i = 0; i < size; i++)
	{
		ns.push_back(normals[i]);
	}
	return ns;
}


bool initOpenGL(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitWindowSize(width,height);
	glutCreateWindow("Showroom");
	GLenum err = glewInit();
	if(err != GLEW_OK)
	{
		exit(0);
	}
}

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

	// Cuda Device setzen
	cudaGLSetGLDevice(cuDev);
	cudaSetDevice(cuDev);
	//cudaFree(0);

	//Speziell für PixelBuffer
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo); //könnte auch arraybuffer sein
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 4, NULL, GL_DYNAMIC_DRAW);//GL_STREAM_READ);
	/*glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);*/

	cudaGraphicsGLRegisterBuffer(&resource, pbo, cudaGraphicsRegisterFlagsNone);

}

void initProjection()
{
	cout << "Generiere die Projektion..." << endl;

	int viewport_width = width;
	int viewport_height = height;

	glViewport(0, 0, width, height);

	float rechts = factor * viewport_width / 2;
	float links = factor * -viewport_width / 2;
	float oben = factor * viewport_height / 2;
	float unten = factor * -viewport_height / 2;

	float zNah = 0.1f;
	float zFern = 450.0f;

	glm::mat4x4 pMatrix = glm::frustum(links, rechts, unten, oben, zNah, zFern);

	int pUniform = glGetUniformLocation(programID, "proj");
	glUniformMatrix4fv(pUniform, 1, GL_FALSE, glm::value_ptr(pMatrix));

	cout << "Projektion fertig generiert!" << endl;
}

void initLights()
{
	cout << "Licht wird generiert!" << endl;

	// Licht 2 - hinten links
	float light2Pos[3] =
	{ -7.0f, 3.0f, -7.0f };
	float light2Ambient[3] =
	{ 0.0f, 0.0f, 0.0f };
	float light2Diffuse[3] =
	{ 0.9f, 0.9f, 0.9f };
	float light2Specular[3] =
	{ 0.8f, 0.8f, 0.8f };

	//Licht 3 - hinten rechts
	float light3Pos[3] =
	{ 7.0f, 3.0f, -7.0f };
	float light3Ambient[3] =
	{ 0.0f, 0.0f, 0.0f };
	float light3Diffuse[3] =
	{ 0.9f, 0.9f, 0.9f };
	float light3Specular[3] =
	{ 0.8f, 0.8f, 0.8f };

	//Licht 4 - vorne links
	float light4Pos[3] =
	{ -7.0f, 3.0f, 7.0f };
	float light4Ambient[3] =
	{ 0.0f, 0.0f, 0.0f };
	float light4Diffuse[3] =
	{ 0.9f, 0.9f, 0.9f };
	float light4Specular[3] =
	{ 0.8f, 0.8f, 0.8f };

	//Licht 5 - vorne rechts
	float light5Pos[3] =
	{ 7.0f, 3.0f, 7.0f };
	float light5Ambient[3] =
	{ 0.0f, 0.0f, 0.0f };
	float light5Diffuse[3] =
	{ 0.9f, 0.9f, 0.9f };
	float light5Specular[3] =
	{ 0.8f, 0.8f, 0.8f };

	//Licht2
	glUniform3f(glGetUniformLocation(programID, "uLightPosition[0]"), light2Pos[0], light2Pos[1],
		light2Pos[2]);
	glUniform3f(glGetUniformLocation(programID, "uLa[0]"), light2Ambient[0], light2Ambient[1],
		light2Ambient[2]);
	glUniform3f(glGetUniformLocation(programID, "uLd[0]"), light2Diffuse[0], light2Diffuse[1],
		light2Diffuse[2]);
	glUniform3f(glGetUniformLocation(programID, "uLs[0]"), light2Specular[0], light2Specular[1],
		light2Specular[2]);

	glUniform3f(glGetUniformLocation(programID, "uLightPosition[1]"), light3Pos[0], light3Pos[1],
		light3Pos[2]);
	glUniform3f(glGetUniformLocation(programID, "uLa[1]"), light3Ambient[0], light3Ambient[1],
		light3Ambient[2]);
	glUniform3f(glGetUniformLocation(programID, "uLd[1]"), light3Diffuse[0], light3Diffuse[1],
		light3Diffuse[2]);
	glUniform3f(glGetUniformLocation(programID, "uLs[1]"), light3Specular[0], light3Specular[1],
		light3Specular[2]);

	glUniform3f(glGetUniformLocation(programID, "uLightPosition[2]"), light4Pos[0], light4Pos[1],
		light4Pos[2]);
	glUniform3f(glGetUniformLocation(programID, "uLa[2]"), light4Ambient[0], light4Ambient[1],
		light4Ambient[2]);
	glUniform3f(glGetUniformLocation(programID, "uLd[2]"), light4Diffuse[0], light4Diffuse[1],
		light4Diffuse[2]);
	glUniform3f(glGetUniformLocation(programID, "uLs[2]"), light4Specular[0], light4Specular[1],
		light4Specular[2]);

	glUniform3f(glGetUniformLocation(programID, "uLightPosition[3]"), light5Pos[0], light5Pos[1],
		light5Pos[2]);
	glUniform3f(glGetUniformLocation(programID, "uLa[3]"), light5Ambient[0], light5Ambient[1],
		light5Ambient[2]);
	glUniform3f(glGetUniformLocation(programID, "uLd[3]"), light5Diffuse[0], light5Diffuse[1],
		light5Diffuse[2]);
	glUniform3f(glGetUniformLocation(programID, "uLs[3]"), light5Specular[0], light5Specular[1],
		light5Specular[2]);

	cout << "Licht erfolgreich generiert!" << endl;
}

void deleteCar()
{
	if (part.size() == 8)
		cout << "Löschen geht nicht! Kein Auto geladen." << endl;
	else
	{
		for (vector<int>::iterator it = vbo.begin() + 8; it != vbo.end(); it++)
		{
			vbo.erase(it);
		}
		for (vector<int>::iterator it = ibo.begin() + 8; it != ibo.end(); it++)
		{
			ibo.erase(it);
		}
		for (vector<int>::iterator it = nbo.begin() + 8; it != nbo.end(); it++)
		{
			nbo.erase(it);
		}
		for (vector<Modell*>::iterator it = part.begin() + 8; it != part.end(); it++)
		{
			part.erase(it);
		}
	}
}

void initBuffers(Modell* m)
{
	float* vertexBuffer = (float*) malloc(sizeof(float) * m->getVertices().size());
	int i = 0;

	for (vector<double>::iterator it = m->getVertices().begin(); it != m->getVertices().end(); it++)
	{
		vertexBuffer[i++] = (float) (*it);
	}
	int* indexBuffer = (int*) malloc(sizeof(int) * m->getIndices().size());
	i = 0;
	for (vector<int>::iterator it = m->getIndices().begin(); it != m->getIndices().end(); it++)
	{
		indexBuffer[i++] = (*it);
	}

	float* normalBuffer = (float*) malloc(sizeof(float) * m->getNormals().size());
	i = 0;

	for (vector<double>::iterator it = m->getNormals().begin(); it != m->getNormals().end(); it++)
	{
		normalBuffer[i++] = (float) (*it);
	}

	glGenVertexArrays(1, &vaoId);
	glBindVertexArray(vaoId);

	glGenBuffers(1, &vboId);
	glBindBuffer(GL_ARRAY_BUFFER, vboId);
	glBufferData(GL_ARRAY_BUFFER, m->getVertices().size() * sizeof(float), vertexBuffer,
		GL_STATIC_DRAW);

	glGenBuffers(1, &iboId);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, iboId);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m->getIndices().size() * sizeof(int), indexBuffer,
		GL_STATIC_DRAW);

	glGenVertexArrays(1, &naoId);
	glBindVertexArray(nboId);

	glGenBuffers(1, &nboId);
	glBindBuffer(GL_ARRAY_BUFFER, nboId);
	glBufferData(GL_ARRAY_BUFFER, m->getNormals().size() * sizeof(float), normalBuffer,
		GL_STATIC_DRAW);



	vbo.push_back(vboId);
	nbo.push_back(nboId);
	ibo.push_back(iboId);
	part.push_back(m);



}

void updateTransformation()
{
	//Model View
	//Translate//
	glm::vec3 translation = glm::vec3(translation_X, translation_Y, zoom);
	glm::mat4x4 mvMatrix = glm::translate(glm::mat4(1.0f), translation);

	//Rotate//
	glm::vec3 rotation = glm::vec3(rotationAngle_X, rotationAngle_Y, 1.0);
	mvMatrix = glm::rotate(mvMatrix, (float) (rotation.y * M_PI / 180), glm::vec3(0.0, 1.0, 0.0));
	mvMatrix = glm::rotate(mvMatrix, (float) (rotation.x * M_PI / 180), glm::vec3(1.0, 0.0, 0.0));

	int mvUniform = glGetUniformLocation(programID, "modelView");
	glUniformMatrix4fv(mvUniform, 1, GL_FALSE, glm::value_ptr(mvMatrix));

	//Normals
	glm::mat4x4 normalMatrix = glm::inverseTranspose(mvMatrix);

	int nUniform = glGetUniformLocation(programID, "normals");
	glUniformMatrix4fv(nUniform, 1, GL_FALSE, glm::value_ptr(normalMatrix));
}

void loadParts(const string &fileName)
{

	ptree Node;


	try
	{
		boost::property_tree::read_json(fileName, Node);

	}
	catch (std::exception const& e)
	{
		std::cerr << e.what() << std::endl;
	}

	//Parsing Part Name
	vector<double> vertices;
	vector<int> indices;
	vector<double> normals;
	vector<double> Ka;
	vector<double> Kd;
	vector<double> Ks;
	string alias = Node.get<string>("alias");
	double Ns = Node.get<float>("Ns");
	double Ni = Node.get<float>("Ni");
	double d = Node.get<float>("d");

	//Parsing modell vertices
	BOOST_FOREACH(boost::property_tree::ptree::value_type &v, Node.get_child("vertices")) {
		vertices.push_back(v.second.get_value<float>());
	}
	BOOST_FOREACH(boost::property_tree::ptree::value_type &v, Node.get_child("indices")) {
		indices.push_back(v.second.get_value<int>());
	}
	BOOST_FOREACH(boost::property_tree::ptree::value_type &v, Node.get_child("Ka")) {
		Ka.push_back(v.second.get_value<float>());
	}
	BOOST_FOREACH(boost::property_tree::ptree::value_type &v, Node.get_child("Kd")) {
		Kd.push_back(v.second.get_value<float>());
	}
	BOOST_FOREACH(boost::property_tree::ptree::value_type &v, Node.get_child("Ks")) {
		Ks.push_back(v.second.get_value<float>());
	}

	normals = calculateNormals(vertices, indices);

	Modell* m = new Modell(alias, vertices, normals, indices, Ni, Ka, d, Kd, Ks,
		Ns);

	initBuffers(m);

}

void initScene()
{
	for (int i = 1; i < 9; i++)
	{
		std::stringstream filename;
		filename << "models\\showroom\\part" << i << ".json";
		cout << "Part" << i << " wird geladen..." << endl;

		partID = i;
		loadParts(filename.str());

	}
}

void initCars(int carID, int anzahl)
{
	string name;
	switch (carID)
	{
	case 1:
		name = "audi_r8";
		break;
	case 2:
		name = "aston_martin";
		break;
	case 3:
		name = "vw_up";
		break;

	case 4:
		name = "teapot";
		break;
	}
	cout << "Lade " << name << " mit " << anzahl << " Teile..." << endl;

	for (int i = 1; i < anzahl + 1; i++)
	{
		std::stringstream s;

		s << "models\\" << name << "\\part" << i << ".json";

		cout << "Part " << i << " wird geladen!" << endl;
		partID = i;

		loadParts(s.str());
	}
}

void createShaderProgram()
{
	cout << "ShaderProgramm wird generiert..." << endl;

	GLuint v = glCreateShader(GL_VERTEX_SHADER);
	GLuint f = glCreateShader(GL_FRAGMENT_SHADER);

	string vString, fString;

	//Lese die Shader aus
	vString = textFileRead("shader\\carConf_VS.glsl");
	fString = textFileRead("shader\\carConf_FS.glsl");
	const char *vertShaderSrc = vString.c_str();
	const char *fragShaderSrc = fString.c_str();

	//Kompiliere VS
	std::cout << "Kompiliere VS." << std::endl;
	glShaderSource(v, 1, &vertShaderSrc, NULL);
	glCompileShader(v);

	char* log = (char*) malloc(1000 * sizeof(char));
	int r;
	glGetShaderInfoLog(v, 1000, &r, log);
	cout << "Vertex Shader Log: " << log << endl;

	//Kompiliere FS
	std::cout << "Kompiliere FS." << std::endl;
	glShaderSource(f, 1, &fragShaderSrc, NULL);
	glCompileShader(f);

	char* fslog = (char*) malloc(1000 * sizeof(char));
	int fs_r;
	glGetShaderInfoLog(f, 1000, &fs_r, fslog);
	cout << "Fragment Shader Log: " << fslog << endl;

	//Linken des ShaderProgramms
	std::cout << "Linking des ShaderProgramms!" << std::endl;
	programID = glCreateProgram();
	glAttachShader(programID, v);
	glAttachShader(programID, f);
	glLinkProgram(programID);

	ATTR_POS = glGetAttribLocation(programID, "vs_in_pos");
	ATTR_NORMAL = glGetAttribLocation(programID, "vs_in_normal");

	cout << "ShaderProgramm ist generiert!" << endl;
}

void processKeyOps()
{
	if (keyStates['1'])
	{
		vw = false;
		initCars(1, 182);
	}
	if (keyStates['2'])
	{
		vw = false;
		initCars(2, 163);
	}
	if (keyStates['3'])
	{
		vw = true;
		initCars(3, 331);
	}
	if (keyStates['4'])
	{
		vw = false;
		initCars(4, 1);
	}


	if (keySpecialStates[GLUT_KEY_UP])
	{
		zoom += 0.3f;
	}
	if (keySpecialStates[GLUT_KEY_DOWN])
	{
		zoom -= 0.3f;
	}
	if (keySpecialStates[GLUT_KEY_LEFT])
	{
		rotationAngle_Y += 15.0f;
	}
	if (keySpecialStates[GLUT_KEY_RIGHT])
	{
		rotationAngle_Y -= 15.0f;
	}
	if (keySpecialStates[GLUT_KEY_END])
	{
		deleteCar();
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
	rotationAngle_Y += 0.45;
	updateTransformation();
	
	for (int i = 0; i < part.size(); i++)
	{

		glUniform3f(glGetUniformLocation(programID, "uKa"), part.at(i)->getKa().at(0), part.at(i)->getKa().at(1), part.at(i)->getKa().at(2));
		glUniform3f(glGetUniformLocation(programID, "uKd"), part.at(i)->getKd().at(0), part.at(i)->getKd().at(1), part.at(i)->getKd().at(2));
		glUniform3f(glGetUniformLocation(programID, "uKs"), part.at(i)->getKs().at(0), part.at(i)->getKs().at(1), part.at(i)->getKs().at(2));
		glUniform1f(glGetUniformLocation(programID, "uNs"), part.at(i)->getNs());
		glUniform1f(glGetUniformLocation(programID, "d"), part.at(i)->getD());
		if (vw)
		{
			if (!part.at(i)->getAlias().compare("GLAS"))
			{
				glUniform1f(glGetUniformLocation(programID, "d"), 0.5f);
			}
			else
			{
				glUniform1f(glGetUniformLocation(programID, "d"), 1.0f);
			}
		}

		
		glBindBuffer(GL_ARRAY_BUFFER, vbo.at(i));
		glVertexAttribPointer(ATTR_POS, 3, GL_FLOAT, false, 0, 0);
		glEnableVertexAttribArray(ATTR_POS);

		glBindBuffer(GL_ARRAY_BUFFER, nbo.at(i));
		glVertexAttribPointer(ATTR_NORMAL, 3, GL_FLOAT, false, 0, 0);
		glEnableVertexAttribArray(ATTR_NORMAL);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo.at(i));

		
		glDrawElements(GL_TRIANGLES, part.at(i)->getIndices().size(), GL_UNSIGNED_INT, 0);
		
	}	

		glFinish();
		//Buffer bei Cuda anmelden

		glReadBuffer(GL_BACK);
		glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
		glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

		cudaError_t r = cudaSuccess;
		CUDA_SAFE_CALLING(cudaMalloc((void**)&d_yuv, arraySize*sizeof(unsigned char)));
		CUDA_SAFE_CALLING(cudaMemset((void*) d_yuv, 127, arraySize * sizeof(unsigned char)));
		CUDA_SAFE_CALLING(cudaGraphicsMapResources(1, &resource, NULL));
		CUDA_SAFE_CALLING(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, NULL, resource));
		callKernel(width,height,d_yuv, devPtr);

		CUDA_SAFE_CALLING(cudaDeviceSynchronize());
		CUDA_SAFE_CALLING(cudaGraphicsUnmapResources(1, &resource, NULL));
		CUDA_SAFE_CALLING(cudaMemcpy( &yuv[0], d_yuv,  yuv.size(), cudaMemcpyDeviceToHost));

		remo->setPicBuf(&yuv[0]);
		remo->encodePB();

		CUDA_SAFE_CALLING(cudaFree((void*)d_yuv));


	glutPostRedisplay();
	glFinish();

	lastTimeMS = st.wMilliseconds;
}

void initCallbacks()
{
	glutDisplayFunc(drawScene);
}

int main(int argc, char** argv)
{
	serverSocket = new UdpSocket();
	serverSocket->Create();
	serverSocket->Bind("192.168.178.50", DEFAULT_PORT+1);

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

	//Initialisiere neuen RemoteEncoder
	glClearColor(0.0, 0.0, 1.0, 1.0);
	glClearDepth(1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glCullFace(GL_BACK);




	//init cuda datenfelder
	devPtr = NULL;
	arraySize = width * height* 1.5;
	yuv = vector<unsigned char>(arraySize); 
	d_yuv = new unsigned char[arraySize];
	
	memset(&yuv[0], 0.0, yuv.size());
	

	createShaderProgram();
	glUseProgram(programID);

	initProjection();

	initLights();

	initScene();

	initCuda();

	glEnableVertexAttribArray(ATTR_POS);
	glEnableVertexAttribArray(ATTR_NORMAL);


	initCallbacks();
	


	

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
