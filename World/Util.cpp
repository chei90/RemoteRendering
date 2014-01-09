// stdafx.h : Includedatei für Standardsystem-Includedateien
// oder häufig verwendete projektspezifische Includedateien,
// die nur in unregelmäßigen Abständen geändert werden.
//
#include "Util.h"



std::string textFileRead(const char *filePath)
{

	std::string content;
	std::ifstream fileStream(filePath, std::ios::in);

	if (!fileStream.is_open())
	{
		std::cerr << "Datei konnte nicht gelesen werden mit Pfad: " << filePath << std::endl;
		return "";
	}

	std::string line = "";
	while (fileStream.good())
	{
		std::getline(fileStream, line);
		content.append(line + "\n");


	}
	fileStream.close();
	return content;
}

int createShaderProgram(const char* vs, const char* fs)
{
	int programId = glCreateProgram();

	int vsId = glCreateShader(GL_VERTEX_SHADER);
	int fsId = glCreateShader(GL_FRAGMENT_SHADER);

	glAttachShader(programId, vsId);
	glAttachShader(programId, fsId);

	std::string t_vsContent = textFileRead(vs);
	std::string t_fsContent = textFileRead(fs);

	const char* vsContent = t_vsContent.c_str();
	const char* fsContent = t_fsContent.c_str();

	std::cout << "FS Source \n " << fsContent << std::endl;
	std::cout << "VS Source \n" << vsContent << std::endl;

	glShaderSource(vsId, 1, &vsContent, NULL);
	glShaderSource(fsId, 1, &fsContent, NULL);

	glCompileShader(vsId);
	glCompileShader(fsId);

	char* e_log = new char[1024];
	GLint compiled;
	glGetShaderiv(vsId, GL_COMPILE_STATUS, &compiled);
	glGetShaderInfoLog(vsId, 1024, 0, e_log);
	std::cout << "\n\n Compiling\n VS: \n" << e_log << std::endl;
	glGetShaderiv(fsId, GL_COMPILE_STATUS, &compiled);
	glGetShaderInfoLog(fsId, 1024, 0, e_log);
	std::cout << "\n FS: \n" << e_log << "\n" << std::endl;

	glBindAttribLocation(programId, Geometry::ATTR_POS, "vs_in_pos");
	glBindAttribLocation(programId, Geometry::ATTR_NORMAL, "vs_in_normal");
	glBindAttribLocation(programId, Geometry::ATTR_COLOR, "vs_in_color");;
	
	std::cout << "Linking Program: \n" << std::endl;
	glLinkProgram(programId);

	return programId;
}

Geometry* createSphere(float r, int n, int k, const char* imageFile)
{
	float dPhi = (float)(2 * M_PI / k);
	float dTheta = (float)(M_PI / n);

	glm::vec4 north = glm::vec4(0.0f, r, 0.0f, 1.0f);
	std::vector<float> vec_vert = std::vector<float>(6 * (n+1) * (k+1));
	float* vertices = new float[6 * (n+1) * (k+1)];
	int height, width = 0;
	float*** image = getImage(imageFile, &height, &width);

	//glm::mat4x4 rot = glm::mat4x4(1.0f);
	//glm::mat4x4 rotX = glm::mat4x4(1.0f);
	//glm::mat4x4 rotY = glm::mat4x4(1.0f);
	glm::vec4 tmp = glm::vec4();
	int counter = 0;

	for(int i = 0; i <= k; i++)
	{
		for(int j = 0; j <= n; j++)
		{
            float u = glm::pi<float>() * i / (float)k;
            float v = 2 * glm::pi<float>() * j / (float)n;
			vertices[counter++] = sin(u) * cos(v);
			vertices[counter++] = sin(u) * sin(v);
			vertices[counter++] = cos(u);
			vertices[counter++] = image[j * height / (n+1)][i * width / (k+1)][0];
			vertices[counter++] = image[j * height / (n+1)][i * width / (k+1)][0];
			vertices[counter++] = image[j * height / (n+1)][i * width / (k+1)][0];

			/*vertices[counter++] = tmp.x;
			vertices[counter++] = tmp.y;
			vertices[counter++] = tmp.z;
			vertices[counter++] = image[j * height / (n+1)][i * width / (k+1)][0];
			vertices[counter++] = image[j * height / (n+1)][i * width / (k+1)][0];
			vertices[counter++] = image[j * height / (n+1)][i * width / (k+1)][0];*/
			//std::cout << "Tmp.x: " << tmp.x << " Tmp.y: " << tmp.y << " Tmp.z: " << tmp.z << std::endl;
		}
	}

	int* indices = new int[2 * (n+2) * (k+1)];
	counter = 0;

	for(int i = 0; i <= k; i++)
	{
		for(int j = 0; j <= n; j++)
		{
			indices[counter++] = (i+1) * (n+1) + j;
			indices[counter++] = i * (n+1) +j;
		}
		indices[counter++] = -1;
	}

	Geometry* sphere = new Geometry();
	sphere->setIndexBuffer(indices, GL_TRIANGLE_STRIP, 2 * (n+2) * (k+1));
	sphere->setVertices(vertices, 6 * (n+1) * (k+1));

	return sphere;
}

float*** getImage(const char* fileName, int* height, int* width)
{
	ilInit();
	ILuint img;
	ilGenImages(1, &img);
	ilBindImage(1);
	ilLoadImage(fileName);
	ILenum error;
	error = ilGetError();

	*height = ilGetInteger(IL_IMAGE_HEIGHT);
	*width = ilGetInteger(IL_IMAGE_WIDTH);

	ILubyte* imgData = ilGetData();

	float*** texImg = new float**[*height];

	for(int i = 0; i < *height; i++)
	{
		texImg[i] = new float*[*width];
		for(int j = 0; j < *width; j++)
		{
			texImg[i][j] = new float[3];
		}
	}
	int k = 0;
	for(int i = 0; i < *height; i++)
	{
		for(int j = 0; j < *width; j++)
		{
			
			texImg[i][j][0] = imgData[k++] / 255.0f;
			texImg[i][j][1] = imgData[k++] / 255.0f;
			texImg[i][j][2] = imgData[k++] / 255.0f;
		}
	}
	
	ilDeleteImages(1, &img);
	std::cout << "Textures loaded!" << std::endl;

	return texImg;
}

void glCheckError(GLenum error, const char* msg)
{
	std::cout << msg << std::endl;
	switch(error)
	{
	case GL_INVALID_ENUM:
		std::cout << "Error: GL_INVALID_ENUM" << std::endl;
		break;
	case GL_INVALID_VALUE:
		std::cout << "Error: GL_INVALID_VALUE" << std::endl;
		break;
	case GL_INVALID_OPERATION:
		std::cout << "Error: GL_INVALID_OPERATION" << std::endl;
		break;
	case GL_STACK_OVERFLOW:
		std::cout << "Error: GL_STACK_OVERFLOW" << std::endl;
		break;
	case GL_STACK_UNDERFLOW:
		std::cout << "Error: GL_STACK_UNDERFLOW" << std::endl;
		break;
	case GL_OUT_OF_MEMORY:
		std::cout << "Error: GL_OUT_OF_MEMORY" << std::endl;
		break;
	case GL_INVALID_FRAMEBUFFER_OPERATION:
		std::cout << "Error: GL_INVALID_FRAMEBUFFER_OPERATION" << std::endl;
		break;
	case GL_TABLE_TOO_LARGE:
		std::cout << "Error: GL_TABLE_TOO_LARGE" << std::endl;
		break;
	case GL_NO_ERROR:
		std::cout << "Error: GL_NO_ERROR" << std::endl;
		break;
	}
	std::cout << "\n" << std::endl;
}
