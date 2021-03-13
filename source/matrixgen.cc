#include <algorithm>
#include <direct.h>

#include "util.h"
#include "matrixgen.h"
#include "sample.h"

#include "mac0.h"

#include "read.h"

#ifdef TBB
#include <tbb/tbb.h>
#endif // TBB

#ifdef OMP
#include <omp.h>
#endif // OMP


void gendiagm(int m, int n, double* d, double* diagm) {
	std::fill(diagm, diagm + m * n, 0);
	int start_idx, s_idx = 0;
	for (int i = 0; i < m; i++)
	{
		start_idx = i * n;
		diagm[start_idx + i] = d[s_idx];
		s_idx++;
	}
}

#ifdef INTERN
void nlvm(int m, int n, double* nlvm, nlvmf nlvm_fn, int vDim, sample_fn s_fn, int thread_ID)
{
	int nlvm_idx = 0;

	std::unique_ptr<double[]> tmp1 = std::make_unique<double[]>(m * vDim);
	std::unique_ptr<double[]> tmp2 = std::make_unique<double[]>(n * vDim);

	nRandomVectorsCreateMKL(m, vDim, tmp1.get(), s_fn, thread_ID);
	nRandomVectorsCreateMKL(n, vDim, tmp2.get(), s_fn, thread_ID);

	for (unsigned int i = 0; i < m; i++)
	{
#if MKL_TEST > 0
		printC1darray(&(tmp1.get()[i * vDim]), vDim, "s1");
#endif // MKL_TEST
		for (unsigned int j = 0; j < n; j++) {
			nlvm_fn(vDim, &(tmp1.get()[i * vDim]), &(tmp2.get()[j * vDim]), &(nlvm[nlvm_idx]));
			nlvm_idx++;
#if MKL_TEST > 0
			printC1darray(&(tmp2.get()[j * vDim]), vDim, "s2");
#endif // MKL_TEST
		}
	}
#if MKL_TEST > 0
	printC2darray(nlvm, m, n);
#endif // MKL_TEST
};
#endif // INTERN

#ifdef EXTERN
void nlvm(int m, int n, double* nlvm, nlvmf nlvm_fn, char const* sample_name, char const* trial_name, char const* id1, char const* id2, int thread_ID)
{
	if (changeFolder(SAMPLING, sample_name, trial_name) != 0)
		return;
	if (!check_if_file_exists(id1))
		return;
	if (!check_if_file_exists(id2))
		return;

	std::vector<std::vector<double>> tmp1_vec = readCSV(id1, thread_ID);
	std::vector<std::vector<double>> tmp2_vec = readCSV(id2, thread_ID);

	int nlvm_idx = 0;
	int vDim = tmp1_vec[0].size(); // = tmp2_vec[0].size() 
	int total_size1 = tmp1_vec.size() * vDim;
	int total_size2 = tmp2_vec.size() * vDim;
	
	std::shared_ptr<double[]> tmp1(new double[total_size1], std::default_delete<double[]>());
	std::shared_ptr<double[]> tmp2(new double[total_size2], std::default_delete<double[]>());

	dVecVecToCArr(tmp1_vec, tmp1.get(), thread_ID);
	dVecVecToCArr(tmp2_vec, tmp2.get(), thread_ID);
	

#ifdef OMP
#pragma omp parallel for default(none) shared(vDim,nlvm,n) firstprivate(nlvm_fn)
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < n; j++) {
			nlvm_fn(vDim, &(tmp1.get()[i * vDim]), &(tmp2.get()[j * vDim]), &(nlvm[i * n + j]), i);
		}
	}
#if MKL_TEST > 0
	printC2darray(nlvm, m, n);
#endif // MKL_TEST
#else
	for (int i = 0; i < m; i++)
	{
#if MKL_TEST > 0
		printC1darray(&(tmp1.get()[i * vDim]), vDim, "s1");
#endif // MKL_TEST
		for (int j = 0; j < n; j++) {
			nlvm_fn(vDim, &(tmp1.get()[i * vDim]), &(tmp2.get()[j * vDim]), &(nlvm[i * n + j]), i);
#if MKL_TEST > 0
			printC1darray(&(tmp2.get()[j * vDim]), vDim, "s2");
#endif // MKL_TEST
		}
	}
#if MKL_TEST > 0
	printC2darray(nlvm, m, n);
#endif // MKL_TEST
#endif // OMP
}
#endif // EXTERN
