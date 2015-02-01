#pragma once

#include <string>
#include <map>

#include "Chameleon.h"

class ConfigFile {
	std::map<std::string,Chameleon> content_;

public:
	ConfigFile(std::string const& configFile);

	Chameleon const& Value(std::string const& section, std::string const& entry) const;

	Chameleon const& Value(std::string const& section, std::string const& entry, double value);
	Chameleon const& Value(std::string const& section, std::string const& entry, std::string const& value);
};
