/*************************************************************************

Copyright 2014 Christoph Eichler

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.                                                                     

*************************************************************************/

#pragma once

#include <string>
#include <vector>



using namespace std;

class Modell
{
	//Data
private:
	string alias;
	vector<double> vertices;
	vector<double> normals;
	vector<int> indices;
	double Ni;
	vector<double> Ka;
	double d;
	vector<double> Kd;
	vector<double> Ks;
	double Ns;

public:
	Modell(string alias,
			vector<double>& vertices,
			vector<double>& normals,
			vector<int>& indices,
			double Ni,
			vector<double>& Ka,
			double d,
			vector<double>& Kd,
			vector<double>& Ks,
			double Ns)
	{
		this->alias = alias;
		this->vertices = vertices;
		this->normals = normals;
		this->indices = indices;
		this->Ka = Ka;
		this->Kd = Kd;
		this->Ks = Ks;
		this->d = d;
		this->Ni = Ni;
		this->Ns = Ns;
	}
	~Modell()
	{
		delete &vertices;
		delete &normals;
		delete &indices;
		delete &Ks;
		delete &Ka;
		delete &Kd;
	}

	string getAlias()
	{
		return alias;
	}

	vector<double>& getVertices()
	{
		return vertices;
	}

	vector<double>& getNormals()
	{
		return normals;
	}

	vector<int>& getIndices()
	{
		return indices;
	}

	vector<double>& getKa()
	{
		return Ka;
	}

	vector<double>& getKs()
	{
		return Ks;
	}

	vector<double>& getKd()
	{
		return Kd;
	}

	double getNi()
	{
		return Ni;
	}

	double getNs()
	{
		return Ns;
	}

	double getD()
	{
		return d;
	}

};
