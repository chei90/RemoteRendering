#include "Main.h"
#include "device_launch_parameters.h"


inline void processKeyOps()
{
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
	}
}

void drawScene(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//GLSHIT here
	glUseProgram(programID);
	glUniformMatrix4fv(modelLocation, 1, false, glm::value_ptr(cam->getView()));
	cam->move(0.04 * moveDir.z, 0.04 * moveDir.x, 0.04 * moveDir.y);
	glm::mat4 viewProj = cam->getProjection() * cam->getView();
	glUniformMatrix4fv(viewProjLoc, 1, false, glm::value_ptr(viewProj));
	earth->draw();


	glFinish();
	//Buffer bei Cuda anmelden
	glReadBuffer(GL_BACK);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);

	RREncode();
	glutSwapBuffers();
	glutPostRedisplay();
	glFinish();
}

void RRKeyCallback(int key, bool pressed)
{
	std::string tmp = pressed ? "pressed" : "released";
	printf("Key %d %s \n", key, tmp);
	if(pressed)
		keyStates[key] = true;
	else
		keyStates[key] = false;
}

void RRMouseDummy(int dx, int dy, int button, int state)
{
	std::string tmp = state == 0 ? "released" : "pressed";
	printf("Pos: %d,%d; Button: %d, %s \n", dx, dy, button, tmp);
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
	glutInitWindowSize(width, height);
	glutCreateWindow("RemoteRenderingServer");
	glewInit();

	glutDisplayFunc(drawScene);

	glViewport(0, 0, width, height);
	glPrimitiveRestartIndex(PRIMITIVE_RESTART);
	glEnable(GL_PRIMITIVE_RESTART);
}

int main(int argc, char** argv)
{
	std::string ip;
	int port; 
	/*
	std::cout << "Insert Ip:" << std::endl;
	std::getline(cin, ip);
	std::cout << "Insert Port: " << std::endl;
	std:cin >> port;
	*/

	//Der Rest wird nicht unterstützt! Noch nicht.
	width = 800;
	height = 600;
	assert(width == 800);
	assert(height == 600);


	RREncoderDesc rdesc;
	rdesc.gfxapi = GL;
	rdesc.w = width;
	rdesc.h = height;
	rdesc.ip = "127.0.0.1";
	rdesc.port = 8081;
	rdesc.keyHandler = RRKeyCallback;
	rdesc.mouseHandler = RRMouseDummy;
 

	initOpenGL(argc, argv);

	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 4, NULL, GL_STATIC_DRAW);

	
	RRInit(rdesc);
	RRWaitForConnection();
	RRSetSource((void*) &pbo);


	programID = createShaderProgram("shader/Main_VS.glsl", "shader/VertexColor_FS.glsl");
	modelLocation = glGetUniformLocation(programID, "model");
	viewProjLoc = glGetUniformLocation(programID, "viewProj");
	cam->move(-5.0f, 0.0f, 0.0f);
	earth = createSphere(1, 64, 32);
	glBindTexture(GL_TEXTURE_2D, createTexture(L"textures/earth.jpg"));
	glActiveTexture(GL_TEXTURE0);
	glUniform1i(glGetUniformLocation(programID, "colorTex"), 0);
	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CCW);
	glCullFace(GL_BACK);
	glClearColor(0.3f, 0, 0, 1.0f);

	
	while(m_continue)
	{
		RRQueryClientEvents();
		glutMainLoopEvent();
		processKeyOps();
	}
	return 0;
}
