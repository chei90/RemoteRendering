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
