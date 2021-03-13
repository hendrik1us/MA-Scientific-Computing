#ifndef WRITE_H
#define WRITE_H

#include "sample.h"
#include "matrixgen.h"

struct NLVM_DIMS
{
	int m;
	int n;
};

struct SAMPLE_DIMS
{
	int m1;
	int m2;
};

void writeSamples(char const* sample_name, char const* trial_name, int vDim, std::vector<SAMPLE_DIMS> sample_dims, sample_fn fn);
void writeSamples(char const* sample_name, char const* trial_name, int vDim, std::vector<SAMPLE_DIMS> sample_dims, sample_fn fn, double scale);
void writeNLVMs(char const* nlvm_name, char const* Exn, char const* sample_name, char const* trial_name, std::vector<NLVM_DIMS> nlvm_dims, nlvmf fn);
void writeResults(char const* nlvm_name, char const* Exn, char const* result_name, char const* sample_name, char const* trial_name, std::vector<NLVM_DIMS> nlvm_dims);
#endif // !WRITE_H





