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
	//glBindAttribLocation(programId, Geometry::ATTR_NORMAL, "vs_in_normal");
	glBindAttribLocation(programId, Geometry::ATTR_COLOR, "vs_in_color");
	glBindAttribLocation(programId, Geometry::ATTR_TEX_COORDS, "vs_in_texCoords");
	
	std::cout << "Linking Program: \n" << std::endl;
	glLinkProgram(programId);

	return programId;
}

Geometry* createSphere(float r, int n, int k)
{
	float dTheta = M_PI / (float)k;
	float dPhi =  2.0f * M_PI / (float)n;

	std::vector<float> vertexInformation = std::vector<float>(8 * (n+1) * (k+1));
	int counter = 0;


	float theta = 0;
	for(int j = 0; j <= k; ++j)
	{

		float sinTheta = sin(theta);
		float cosTheta = cos(theta);
		float phi = 0;

		for(int i = 0; i <= n; ++i)
		{
			float sinPhi = sin(phi);
			float cosPhi = cos(phi);

			vertexInformation[counter++] = r * sinTheta * cosPhi;
			vertexInformation[counter++] = r * cosTheta;
			vertexInformation[counter++] = r * sinTheta * sinPhi;
			vertexInformation[counter++] = sinTheta * cosPhi;
			vertexInformation[counter++] = cosTheta;
			vertexInformation[counter++] = sinTheta * sinPhi;
			vertexInformation[counter++] = i / (float)n;
			vertexInformation[counter++] = j / (float)k;

			phi += dPhi;
		}
		theta += dTheta;
	}

	std::vector<int> indexInformation = std::vector<int>(k * (2 * (n+1) + 1));
	counter = 0;

	for(int j = 0; j < k; ++j)
	{
		for(int i = 0; i <= n; ++i)
		{
			indexInformation[counter++] = (j+1) * (n+1) + i;
			indexInformation[counter++] = j * (n+1) + i;
		}
		indexInformation[counter++] = PRIMITIVE_RESTART;
	}

	Geometry* sphere = new Geometry();
	sphere->setIndexBuffer(&indexInformation[0], GL_TRIANGLE_STRIP, 2 * (n+2) * (k+1));
	sphere->setVertices(&vertexInformation[0], 8 * (n+1) * (k+1));

	return sphere;
}

GLuint createTexture(const wchar_t* fileName)
{
	GLuint texID;
	int width, height, imgFormat, internalFormat = 0;
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
	glBindTexture(GL_TEXTURE_2D, texID);
	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, imgFormat, GL_FLOAT, (void*) imgData);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);
    glCheckError(glGetError(), "createTexture");
	return texID;
}

float* getImage(const wchar_t* fileName, int* height, int* width, int* imgFormat)
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
	*imgFormat = ilGetInteger(IL_IMAGE_FORMAT);

	ILubyte* imgData = ilGetData();

	float* texImg = new float[*height * *width * 3];
	for(int i = 0; i < *height * *width * 3; i++)
	{
		texImg[i] = imgData[i] / 255.0f;
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
