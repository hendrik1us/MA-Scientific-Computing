#include "read.h"

#include <fstream>
#include <sstream>
#include <direct.h>
#include <algorithm>

/* Split */
template <typename Out>
void split(const std::string& s, char delim, Out result) {
	std::istringstream iss(s);
	std::string item;
	while (std::getline(iss, item, delim)) {
		*result++ = item;
	}
}

std::vector<std::string> split(const std::string& s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, std::back_inserter(elems));
	return elems;
}

/* Read CSV Files */
std::vector<std::vector<double>> readCSV(char const* id, int thread_ID) {
	std::fstream data;
	std::string curr, tmp;
	std::vector<std::string> dataStrVec;

	data.open(id, std::ios::in);
	if (data.is_open()) {   //checking whether the file is open     
		while (std::getline(data, tmp)) {  //read data from file object and put it into string.
			dataStrVec.push_back(tmp);   //print the data of the string
		}
		data.close();
	}
	std::vector<std::vector<double>> dataMat;
	std::vector<std::string> tmpStrVec;
	char delim = ',';
	for (unsigned int i = 0; i < dataStrVec.size(); i++)
	{
		tmpStrVec = split(dataStrVec[i], delim);
		std::vector<double> tmpVec(tmpStrVec.size());
		std::transform(tmpStrVec.begin(), tmpStrVec.end(), tmpVec.begin(), [](const std::string& val)
			{
				return std::stod(val);
			});
		dataMat.push_back(tmpVec);
	}
	std::cout << "		Data was loaded" << std::endl;
	return dataMat;
}


/* Auxiliary */
void dVecVecToCArr(std::vector<std::vector<double>> dVecVec, double* ret, int thread_ID) {

	int total_size = 0;
	for (auto& vec : dVecVec)
		total_size += vec.size();

	std::vector<double> flattened;
	flattened.reserve(total_size);

	for (auto& vec : dVecVec)
		for (auto& elem : vec)
			flattened.push_back(elem);

	std::copy(flattened.begin(), flattened.end(), ret);
}