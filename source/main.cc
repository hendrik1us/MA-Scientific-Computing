/* This code is supposed to check the theoretical base developed in the paper "Why Are Big Data Matrices
Approximately Low Rank?" by Madeleine Udell and Alex Townsend with numerical simulations */

#include <algorithm>
#include <iostream>
#include <chrono>
#include <numeric>
#include <direct.h>

#include <mac0.h>

#ifdef TBB
#include <tbb/tbb.h>
#endif // TBB

#ifdef OMP
#include <omp.h>
#endif // !OMP

#ifdef MKL
#include <mkl.h>
#endif // MKL

#include "util.h"
#include "read.h"
#include "write.h"
#include "sample.h"
#include "mathfns.h"
#include "matrixgen.h"

//#pragma optimize("", off)

#ifdef _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif // _CRTDBG_MAP_ALLOC

int main()
{
#ifdef DEBUG
	_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDOUT);
	_CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDOUT);
	_CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDOUT);
#endif // DEBUG

		int factor = 100;
		std::vector<int> ns1(30);
		std::vector<int> ns2(30);
		std::iota(std::begin(ns1), std::end(ns1), 1);
		std::iota(std::begin(ns2), std::end(ns2), 1);

		std::transform(ns1.begin(), ns1.end(), ns1.begin(), [factor](int& c) { return c * factor; });
		std::transform(ns2.begin(), ns2.end(), ns2.begin(), [factor](int& c) { return c * factor; });

		std::vector<NLVM_DIMS> nlvm_dims;
		std::vector<SAMPLE_DIMS> sample_dims;

		for (int i = 0; i < ns1.size(); i++)
		{
			NLVM_DIMS dims1{ ns1[i],ns2[i] };
			nlvm_dims.push_back(dims1);
			SAMPLE_DIMS dims2{ ns1[i],ns2[i] };
			sample_dims.push_back(dims2);
		}

		/* Start Processing */
		auto start = std::chrono::high_resolution_clock::now();
		
		/* Generate samples from 1000-dimensional unit sphere für 5 trials. This is the sample base for all examples.*/
		writeSamples("UnifSampleFromNSphere", "Trial1", 1001, sample_dims, unifSampleFromNSphereMKL);
		writeSamples("UnifSampleFromNSphere", "Trial2", 1001, sample_dims, unifSampleFromNSphereMKL);
		writeSamples("UnifSampleFromNSphere", "Trial3", 1001, sample_dims, unifSampleFromNSphereMKL);
		writeSamples("UnifSampleFromNSphere", "Trial4", 1001, sample_dims, unifSampleFromNSphereMKL);
		writeSamples("UnifSampleFromNSphere", "Trial5", 1001, sample_dims, unifSampleFromNSphereMKL);

		/* From now on: Take f = linear kernel function as an example */
		
		/* Generate samples from nlvm based on f for 5 trials */
		writeNLVMs("LINEARK", "Ex1", "UnifSampleFromNSphere", "Trial1", nlvm_dims, linearKMKL);
		writeNLVMs("LINEARK", "Ex1", "UnifSampleFromNSphere", "Trial2", nlvm_dims, linearKMKL);
		writeNLVMs("LINEARK", "Ex1", "UnifSampleFromNSphere", "Trial3", nlvm_dims, linearKMKL);
		writeNLVMs("LINEARK", "Ex1", "UnifSampleFromNSphere", "Trial4", nlvm_dims, linearKMKL);
		writeNLVMs("LINEARK", "Ex1", "UnifSampleFromNSphere", "Trial5", nlvm_dims, linearKMKL);
		
		/* Calculate epsilon ranks based on the generated nlvm sample matrices for 5 trials */
		writeResults("LINEARK", "Ex1", "EpsRank", "UnifSampleFromNSphere","Trial1",nlvm_dims);
		writeResults("LINEARK", "Ex2", "EpsRank", "UnifSampleFromNSphere", "Trial1", nlvm_dims);
		writeResults("LINEARK", "Ex3", "EpsRank", "UnifSampleFromNSphere", "Trial1", nlvm_dims);
		writeResults("LINEARK", "Ex4", "EpsRank", "UnifSampleFromNSphere", "Trial1", nlvm_dims);
		writeResults("LINEARK", "Ex5", "EpsRank", "UnifSampleFromNSphere", "Trial1", nlvm_dims);
		
		auto stop = std::chrono::high_resolution_clock::now();
		/* Finished processing */

		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << "Time taken by function: "
			<< duration.count() << " microseconds" << std::endl;
		std::cout << "That are " << duration.count() / 1000000 << " seconds" << std::endl;

#ifdef DEBUG
	int leak = _CrtDumpMemoryLeaks();
	_ASSERTE(_CrtCheckMemory());	
#endif // DEBUG
}