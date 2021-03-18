/* This code is supposed to check the theoretical base developed in the paper "Why Are Big Data Matrices
Approximately Low Rank?" by Madeleine Udell and Alex Townsend with numerical simulations */

#include <algorithm>
#include <iostream>
#include <chrono>
#include <mac0.h>
#include <omp.h>
//#include <tbb/tbb.h>

#ifdef OPENMP
#include <omp.h>
#endif // OPENMP

#include "util.h"
#include "read.h"
#include "sample.h"
#include "mathfns.h"
#include "matrixgen.h"

//#pragma optimize("", off)

#ifdef MKL
#include <mkl.h>
#endif // MKL

#include <numeric>

#ifdef TBB
#include "write.h"
#else
#ifdef _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif // _CRTDBG_MAP_ALLOC
#endif // TBB

#include "write.h"

#include <numeric>

#ifdef TBB
void ParallelApplyFoo(size_t n) {
	std::shared_ptr<double[]> U(new double[100000]);
	//std::weak_ptr<double[]> weak = U;
	tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
		[=](const tbb::blocked_range<size_t>& r) {
			for (size_t i = r.begin(); i != r.end(); ++i) {
				unifSampleFromNBallMKL(100000, U.get(), 0);
			}
		}
	);
}
#endif

void ParallelApplyFooOMP(size_t n) {
	std::shared_ptr<double[]> U(new double[100000]);
	//std::weak_ptr<double[]> weak = U;
#pragma omp parallel for
	for (int i = 0; i < 40; ++i) {
		unifSampleFromNBallMKL(100000, U.get(), 0);
	}

}

#include <direct.h>
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

		std::vector<int> ns1(10);
		std::vector<int> ns2(10);
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

		auto start = std::chrono::high_resolution_clock::now();
		
		
		/*
		writeNLVMs("POLYNOMIAL3K", "Ex5", "UnifSampleFromNSphere", "Trial1", nlvm_dims, polynomiald3KMKL);
		writeNLVMs("POLYNOMIAL3K", "Ex5", "UnifSampleFromNSphere", "Trial2", nlvm_dims, polynomiald3KMKL);
		writeNLVMs("POLYNOMIAL3K", "Ex5", "UnifSampleFromNSphere", "Trial3", nlvm_dims, polynomiald3KMKL);
		writeNLVMs("POLYNOMIAL3K", "Ex5", "UnifSampleFromNSphere", "Trial4", nlvm_dims, polynomiald3KMKL);
		writeNLVMs("POLYNOMIAL3K", "Ex5", "UnifSampleFromNSphere", "Trial5", nlvm_dims, polynomiald3KMKL);
		*/

		/*writeResults("01LOSSMKL", "Ex7", "EpsRank", "UnifSampleFromNSphere","Trial1",nlvm_dims);
		writeResults("01LOSSMKL", "Ex7", "EpsRank", "UnifSampleFromNSphere", "Trial2", nlvm_dims);
		*/
		//writeResults("01LOSSMKL", "Ex7", "EpsRank", "UnifSampleFromNSphere", "Trial3", nlvm_dims);
		//writeResults("01LOSSMKL", "Ex7", "EpsRank", "UnifSampleFromNSphere", "Trial4", nlvm_dims);		
		
		/*writeResults("POLYNOMIAL3K", "Ex5", "EpsRank", "UnifSampleFromNSphere", "Trial2", nlvm_dims);
		writeResults("POLYNOMIAL3K", "Ex5", "EpsRank", "UnifSampleFromNSphere", "Trial3", nlvm_dims);
		writeResults("POLYNOMIAL3K", "Ex5", "EpsRank", "UnifSampleFromNSphere", "Trial4", nlvm_dims);
		writeResults("POLYNOMIAL3K", "Ex5", "EpsRank", "UnifSampleFromNSphere", "Trial5", nlvm_dims);
		*/
	
		//writeSamples("UnifSampleFromN2Ball", "Trial1", 1001, sample_dims, unifSampleFromN2BallMKL);
		/*writeSamples("UnifSampleFromN2Sphere", "Trial2", 1001, sample_dims, unifSampleFromN2SphereMKL);
		writeSamples("UnifSampleFromN2Sphere", "Trial3", 1001, sample_dims, unifSampleFromN2SphereMKL);
		writeSamples("UnifSampleFromN2Sphere", "Trial4", 1001, sample_dims, unifSampleFromN2SphereMKL);
		writeSamples("UnifSampleFromN2Sphere", "Trial5", 1001, sample_dims, unifSampleFromN2SphereMKL);
		*/

		//writeNLVMs("COSINEK", "Ex4", "UnifSampleFromN2Sphere", "Trial1", nlvm_dims, cosineKMKL);
		/*writeNLVMs("COSINEK", "Ex4", "UnifSampleFromN2Sphere", "Trial3", nlvm_dims, cosineKMKL);
		writeNLVMs("COSINEK", "Ex4", "UnifSampleFromN2Sphere", "Trial4", nlvm_dims, cosineKMKL);
		writeNLVMs("COSINEK", "Ex4", "UnifSampleFromN2Sphere", "Trial5", nlvm_dims, cosineKMKL);
		*/

	    /*writeResults("COSINEK", "Ex4", "EpsRank", "UnifSampleFromN2Sphere", "Trial2", nlvm_dims);
		writeResults("COSINEK", "Ex4", "EpsRank", "UnifSampleFromN2Sphere", "Trial3", nlvm_dims);
		writeResults("COSINEK", "Ex4", "EpsRank", "UnifSampleFromN2Sphere", "Trial4", nlvm_dims);
		writeResults("COSINEK", "Ex4", "EpsRank", "UnifSampleFromN2Sphere", "Trial5", nlvm_dims);
		*/
		

		auto stop = std::chrono::high_resolution_clock::now();

		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

		std::cout << "Time taken by function: "
			<< duration.count() << " microseconds" << std::endl;
		std::cout << "That are " << duration.count() / 1000000 << " seconds" << std::endl;


#ifdef DEBUG
	
	int leak = _CrtDumpMemoryLeaks();
	_ASSERTE(_CrtCheckMemory());
	
#endif // DEBUG

}





#if 0

	std::string buffer(EXAMPLE_PATH);
	buffer.append("Ex1\\Trial1");
	chdir(buffer.c_str());

	std::vector<std::vector<double>> tmp = readCSV("1MA_Dat_Ex1_SED_UnifSampleFromNBall_Trial1.csv", 0);
	int m = tmp.size(), n = tmp[0].size();
	std::shared_ptr<double[]> nlvm(new double[m * n], std::default_delete<double[]>());

	dVecVecToCArr(tmp, nlvm.get(), 0);

	int no_conv_flag = 0;
	unsigned int epsRank = getEpsRank(m, n, nlvm.get(), 0.01, 0.01, no_conv_flag);


	//writeSamples("UnifSampleFromNSphere", "Trialx", 1000, ns1, ns2, unifSampleFromNSphereMKL);


	//writeNLVMs("SED", "Ex1", "UnifSampleFromNBall", "Trial5", nlvm_dims, sedMKL);
	//writeNLVMs("SED", "Ex1", "UnifSampleFromNSphere", "Trial5", nlvm_dims, sedMKL);
#endif // 0










#if 0

int a_mrows, a_ncols;
double* bi, * d, * e, * tauq, * taup;

a_mrows = 3;
a_ncols = 3;

minmn = std::min(a_mrows, a_ncols);

bi = new double[a_mrows * a_ncols];
d = new double[minmn];
e = new double[minmn - 1];
tauq = new double[minmn];
taup = new double[minmn];

memcpy(bi, a, a_mrows* a_ncols * sizeof(double));

/* General matrix to bidiagonal form */
gm2bdf(a_mrows, a_ncols, bi, d, e, tauq, taup);

printC2darray(bi, a_mrows, a_ncols, "bi");
printC1darray(d, minmn, "d");
printC1darray(e, minmn - 1, "e");
printC1darray(tauq, minmn, "tauq");
printC1darray(taup, minmn, "taup");

/* Get orthogonal matrices Q and P^T formed by gm2bdf*/
char const idQ = 'Q';
char const idP = 'P';
int q_mrows, q_ncols, pt_mrows, pt_ncols;
double* Q, * PT;

q_mrows = a_mrows;
q_ncols = a_mrows;
pt_mrows = a_ncols;
pt_ncols = a_ncols;

Q = new double[a_mrows * a_ncols];
PT = new double[a_mrows * a_ncols];
//?
memcpy(Q, bi, a_mrows* a_ncols * sizeof(double));
memcpy(PT, bi, a_mrows* a_ncols * sizeof(double));

#if 0
double* tauparr = new double[a_mrows * a_ncols];
double* tauqarr = new double[a_mrows * a_ncols];
int start_idx;
std::fill(tauparr, tauparr + a_mrows * a_ncols, 0);
std::fill(tauqarr, tauqarr + a_mrows * a_ncols, 0);
for (int i = 0; i < a_mrows; i++)
{
	start_idx = i * a_ncols;
	if (a_mrows >= a_ncols)
	{
		if (i > 1)
		{
			for (int j = start_idx; j < start_idx + i - 1; j++)
			{
				tauqarr[j] = bi[j];
			}
		}
		for (int j = start_idx + i + 1; j < start_idx + a_ncols; j++)
		{
			tauparr[j] = bi[j];
		}

	}
	else
	{
		if (i > 1)
		{
			for (int j = start_idx; j < start_idx + i - 1; j++)
			{
				tauqarr[j] = bi[j];
			}
		}
		for (int j = start_idx + i + 1; j < start_idx + a_ncols; j++)
		{
			tauparr[j] = bi[j];
		}
	}

}

printC2darray(tauqarr, a_mrows, a_ncols, "tauqarr");
printC2darray(tauparr, a_mrows, a_ncols, "tauparr");

#endif


// dimensions of orthogonal matrix to be returned: om_mrow, om_ncols

getombygebrd(idQ, q_mrows, q_ncols, a_mrows, a_ncols, Q, tauq);

printC2darray(Q, q_mrows, q_ncols, "Q");

getombygebrd(idP, pt_mrows, pt_ncols, a_mrows, a_ncols, PT, taup); //?

printC2darray(PT, pt_mrows, pt_ncols, "P");

#ifdef MKL_TEST

double* b = new double[a_mrows * a_ncols];
int start_idx = 0;
std::fill(b, b + a_mrows * a_ncols, 0);
// b is upper bidiagonal matrix
int e_idx = 0;
for (int i = 0; i < a_mrows; i++)
{
	start_idx = i * a_ncols;
	b[start_idx + i] = d[i];
	if (a_mrows >= a_ncols)
	{
		if (start_idx + i + 1 < start_idx + a_ncols) {
			b[start_idx + i + 1] = e[e_idx];
			e_idx++;
		}
	}
	else
	{
		if (start_idx + i - 1 > -1) {
			b[start_idx + i - 1] = e[e_idx];
			e_idx++;
		}
	}

}






double* res1 = new double[a_mrows * a_ncols];
double* res2 = new double[a_mrows * a_ncols];

memcpy(res1, b, a_mrows * a_ncols * sizeof(double));
memcpy(res2, b, a_mrows * a_ncols * sizeof(double));

/* Q*B */
LAPACKE_dormbr(LAPACK_ROW_MAJOR, 'Q', 'L', 'N',
	a_mrows, a_ncols, a_ncols, bi, a_ncols, tauq, res1, a_ncols);


/* QB*PT */
LAPACKE_dormbr(LAPACK_ROW_MAJOR, 'P', 'R', 'N',
	a_mrows, a_ncols, a_mrows, bi, a_ncols, tauq, res2, a_ncols);

mamupliMKL(a_mrows, a_ncols, a_ncols, res1, res2, res2);
printC2darray(res2, a_mrows, a_ncols, "res2");




double* c = new double[a_mrows * a_ncols];
std::fill(c, c + a_mrows * a_ncols, 0);

printC2darray(b, a_mrows, a_ncols, "B");

mamupliMKL(q_mrows, q_ncols, a_ncols, Q, b, c); //Q*B
printC2darray(c, q_mrows, a_ncols, "QB"); // ergibt original matrix

mamupliMKL(q_mrows, a_ncols, pt_ncols, c, PT, c); //QB*PT
printC2darray(c, q_mrows, pt_ncols, "Q*B*PT"); //Ergebnis falsch

/* SVD */
double* rsv = new double[a_ncols * a_ncols];
double* lsv = new double[a_mrows * a_mrows];
std::fill(rsv, rsv + a_ncols * a_ncols, 0);
std::fill(lsv, lsv + a_mrows * a_mrows, 0);
std::fill(c, c + a_mrows * a_ncols, 0);

int nrsv = a_ncols;
int nlsv = a_mrows;

char const bi_id = 'L';

#if 1
std::cout << "\nPerform SVD" << std::endl;
getsvdfbidiagm(bi_id, a_mrows, nrsv, nlsv, d, e, Q, PT, c);

printC1darray(d, minmn, "s");
printC2darray(Q, a_ncols, a_ncols, "rsv");
printC2darray(PT, a_mrows, a_mrows, "lsv");

std::cout << "\nReconstruct:" << std::endl;

double* sdiag = new double[a_mrows * a_ncols];
std::fill(sdiag, sdiag + a_mrows * a_ncols, 0);
int s_idx = 0;
for (int i = 0; i < a_mrows; i++)
{
	start_idx = i * a_ncols;
	sdiag[start_idx] = d[s_idx];
	sdiag++;
}
mamupliMKL(q_mrows, q_ncols, a_ncols, Q, d, c);
mamupliMKL(a_mrows, a_ncols, a_ncols, c, PT, c);
printC2darray(c, a_mrows, a_ncols, "recon");



#endif

#endif
#if 0
LAPACKE_dbdsdc(LAPACK_ROW_MAJOR, 'L', 'I', a_mrows,
	d, e, lsv, lapack_int ldu, double* vt, lapack_int ldvt, double* q,
	lapack_int * iq);
#endif


#endif // MKL_TEST





#if 0
SampleGenerator SG;
SG.setProperties(100, 100, usfnsMKL);

double* sample = SG.getSample();

double* a = SG.getNSamples();
/*for (size_t i = 0; i < 10000; i++)
{
	std::cout << ar[i] << " ";
}
*/


//arma::arma_rng::set_seed_random();


std::cout << "MKL Test " << std::endl;
const int n_rows = 100, n_cols = 100;
int lda = n_cols;
double d[n_rows], e[n_rows - 1], tauq[n_rows], taup[n_rows];


int status = LAPACKE_dgebrd(LAPACK_ROW_MAJOR, n_rows, n_cols, a, lda, d, e, tauq, taup);  // A = Q B P^T

char id = 'Q';
int n_rows_Q = n_rows;
int n_cols_Q = n_cols;
int size = n_rows;
double q[n_rows];
memcpy(q, a, size * sizeof(double));
status = LAPACKE_dorgbr(LAPACK_ROW_MAJOR, id, n_rows_Q, n_cols_Q,
	n_cols, q, n_rows, tauq);

id = 'P';
double* p;
memcpy(p, a, size * sizeof(double));
status = LAPACKE_dorgbr(LAPACK_ROW_MAJOR, id, n_rows_Q, n_cols_Q,
	n_cols, p, n_rows, taup);

char uplo = 'U';
int nRightSingVals = n_rows;
int nLeftSingVals = n_rows;
int ncc = 0, ldc = 0;
double* c = 0;
int order = n_rows; //???
status = LAPACKE_dbdsqr(LAPACK_ROW_MAJOR, uplo, order, nRightSingVals,
	nLeftSingVals, ncc, d, e, p, nRightSingVals,
	q, n_cols, c, ldc);


double* res1;
double alpha = 1, beta = 0;
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_rows,
	n_cols, n_rows, alpha, q, n_rows, d, n_cols, beta, res1,
	n_cols);

double* res2;
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_rows,
	n_cols, n_rows, alpha, res1, n_rows, p, n_cols, beta, res2,
	n_cols);

for (size_t i = 0; i < n_rows * n_cols; i++)
{
	std::cout << a[i] - res2[i] << " ";
}

#endif