#ifndef READ_H
#define READ_H

#include <iostream>
#include <vector>

#include "mac0.h"

/* Read CSV Files */
std::vector<std::vector<double>> readCSV(char const* id, int thread_ID);

/* Auxiliary */
#ifdef MKL
void dVecVecToCArr(std::vector<std::vector<double>> dVecVec, double* ret, int thread_ID);
#endif // MKL

#endif // !READ_H
