/*
 * Modell.hpp
 *
 *  Created on: 29.07.2013
 *      Author: christoph
 */

#ifndef MODELL_H_
#define MODELL_H_

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

#endif /* MODELL_HPP_ */
