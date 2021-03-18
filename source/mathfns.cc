#include "mac0.h"


#ifdef MKL
#include <mkl.h>
#include <algorithm>
#include <memory>

#include "matrixgen.h"
#endif // MKL

#include "util.h"
#include "mathfns.h"

#include <numeric>
#include <iterator>

#define _USE_MATH_DEFINES
#include <math.h>

#include <vector>

#ifdef OMP
#include <omp.h>
#endif // OMP

constexpr double mone = -1.0;

/* SVD */
#ifdef MKL
GM2BDF_RESULT gm2bdf(int a_mrows, int a_ncols, double* a, double* d, double* e,
	double* tauq, double* taup) {

	int layout, lda, status;

	layout = LAPACK_ROW_MAJOR;
	lda = a_ncols;

	status = LAPACKE_dgebrd(layout, a_mrows, a_ncols, a, lda, d, e, tauq, taup);
	GM2BDF_RESULT tmp{ status };
	return tmp;
}

GETOMBYGEBRD_RESULT getombygebrd(char const id, int mrows, int ncols, int origm_mrows, int origm_ncols, double* a, double* tau) {
	int layout, k, lda, status;

	layout = LAPACK_ROW_MAJOR;
	lda = ncols;

	if (id == 'Q')
		k = origm_ncols;
	else
		k = origm_mrows;

	status = LAPACKE_dorgbr(layout, id, mrows, ncols, k, a, lda, tau);
	GETOMBYGEBRD_RESULT tmp{ status };
	return tmp;
}

GETSVDFBIDIAGM_RESULT getsvdfbidiagm(char const id, int b_size, int nRightSingVecs, int nLeftSingVecs,
	double* singVals_fromd, double* secDiag_frome, double* rightSingVecs, double* leftSingVecs, double* c)
{

	int layout, ncc, ldvt, ldu, ldc, status;

	layout = LAPACK_ROW_MAJOR;
	ncc = 0;
	ldvt = nRightSingVecs;
	ldu = b_size;
	ldc = 0;

	status = LAPACKE_dbdsqr(layout, id, b_size, nRightSingVecs, nLeftSingVecs, ncc, singVals_fromd, secDiag_frome, rightSingVecs, ldvt,
		leftSingVecs, ldu, c, ldc);
	GETSVDFBIDIAGM_RESULT tmp{ status };
	return tmp;
}

int svdMKL(int m, int n, double* A, double* U, double* s, double* VT, double* superb) {

	int layout, lda, ldu, ldvt, status;
	char const jobu = 'A', jobvt = 'A';

	layout = LAPACK_ROW_MAJOR;
	lda = n;
	ldu = m;
	ldvt = n;

	return LAPACKE_dgesvd(layout, jobu, jobvt, m, n, A, lda, s, U, ldu, VT, ldvt, superb);
}

/* Rank-k-Approximation by SVD */
void mreconMKL(int m, int n, int k, double* U, double* S, double* VT, double* A, double* TMP, int thread_ID) {

	int ldU, ldS, ldUS, ldVT;

	ldU = m;
	ldS = n;
	ldUS = n;
	ldVT = n;
	if (k == 0)
		k++;

	mamupliMKL(m, k, n, ldU, U, ldS, S, TMP, thread_ID);
#if MKL_TEST > 0
	printC2darray(TMP.get(), m, n, "U*S");
#endif
	mamupliMKL(m, k, n, ldUS, TMP, ldVT, VT, A, thread_ID);
#if MKL_TEST > 0
	printC2darray(A, m, n, "U*S*VT");
#endif
}

/* Miscellaneous */
double getewlavMKL(int size, double* arr, int thread_ID) {

	int idx, inc = 1;

	idx = cblas_idamax(size, arr, inc);
	return arr[idx];
}

double getmdiffMKL(int size, double* A, double* B, int thread_ID) {

	int idx, inc;
	double mdiff, ret;

	inc = 1;
	std::unique_ptr<double[]> diff = std::make_unique<double[]>(size);
	vdSub(size, A, B, diff.get());
	mdiff = getewlavMKL(size, diff.get(), thread_ID);
	vdAbs(1, &mdiff, &ret);
	return ret;
}

void mamupliMKL(int a_mrows, int a_ncols, int b_ncols, int lda, double* a, int ldb, double* b, double* c, int thread_ID) {

	CBLAS_LAYOUT layout;
	CBLAS_TRANSPOSE transa, transb;
	int ldc;
	double alpha, beta;

	layout = CblasRowMajor;
	transa = CblasNoTrans;
	transb = CblasNoTrans;
	alpha = 1.0;
	beta = 0.0;
	ldc = b_ncols;

	/* Perform matrix multiplication a * b = c */
	cblas_dgemm(layout, transa, transb, a_mrows, b_ncols, a_ncols, alpha, a, lda, b, ldb, beta, c, ldc);
}

void mamupliMKL(int a_mrows, int a_ncols, int b_ncols, double* a, double* b, double* c) {
	mamupliMKL(a_mrows, a_ncols, b_ncols, a_ncols, a, b_ncols, b, c, 0);
}

void linearKMKL(int size, double* a, double* b, double* c, int thread_ID) {
	int inc = 1;
	*c = cblas_ddot(size, a, inc, b, inc);
}

void sedMKL(int size, double* a, double* b, double* c, int thread_ID) {
	int inc = 1;
	std::unique_ptr<double[]> tmp = std::make_unique<double[]>(size);
	vdSub(size, a, b, tmp.get());
	*c = cblas_ddot(size, tmp.get(), inc, tmp.get(), inc);
}

void polynomiald2KMKL(int size, double* a, double* b, double* c, int thread_ID) {
	double pow = 2.0;
	double offset = 1;
	int inc = 1;
	int vsize = 1;
	double tmp;
	tmp = cblas_ddot(size, a, inc, b, inc);
	tmp = tmp + offset;
	vdPowx(vsize, &tmp, pow, c);
}
void polynomiald3KMKL(int size, double* a, double* b, double* c, int thread_ID) {
	double pow = 3.0;
	double offset = 1;
	int inc = 1;
	int vsize = 1;
	double tmp;
	tmp = cblas_ddot(size, a, inc, b, inc);
	tmp = tmp + offset;
	vdPowx(vsize, &tmp, pow, c);
}

void cosineKMKL(int size, double* a, double* b, double* c, int thread_ID) {
	int inc = 1;
	int vsize = 1;
	double tmp = M_PI_2 * cblas_ddot(size, a, inc, b, inc);
	vdCos(vsize, &tmp, c);
	*c = M_PI_4 * *c;
}

void bsplineMKL(int vsize, double* a, double deg, double* c, int thread_ID) {
	/* not used */
	double sign = 0.0;
	double tmp = 0.0;
	*c = 0.0;
	for (int r = 0; r <= deg + 1; r++)
	{
		vdPowx(vsize, &mone, r, &sign);
		tmp = *a + (deg + 1) / 2 - r;
		vdPowx(vsize, &tmp, deg, &tmp);
		tmp = std::max(0.0, tmp);
		*c = sign * (factorial(deg + 1) / (factorial(r) * factorial(deg + 1 - r))) * tmp + *c;
	}
	*c = (1.0 / factorial(deg)) * *c;
}

void bsplineKMKL(int size, double* a, double* b, double* c, int thread_ID) {
	/* not used */
	int vsize = 1;
	double n = 2.0;
	double deg = 2.0 * n + 1.0;
	double diff;
	std::unique_ptr<double[]> bspl1D = std::make_unique<double[]>(size);
	*c = 1.0;
	for (unsigned int i = 0; i < size; i++)
	{
		diff = a[i] - b[i];
		bsplineMKL(vsize, &diff, deg, &(bspl1D.get()[i]), thread_ID);
		*c = bspl1D.get()[i] * *c;
	}
}

void _01LossMKL(int size, double* a, double* b, double* c, int thread_ID) {
	int vsize = 1;
	double tmp = 0;
	double delta = 1.4;
	sedMKL(size, a, b, &tmp, thread_ID);
	vdSqrt(vsize, &tmp, &tmp);
	if (tmp < delta)
		*c = 0;
	else
		*c = 1;
}

void l1LossMKL(int size, double* a, double* b, double* c, int thread_ID) {
	int inc = 1;
	std::unique_ptr<double[]> diff = std::make_unique<double[]>(size);
	vdSub(size, a, b, diff.get());
	*c = cblas_dasum(size, diff.get(), inc);
}

void chisqrKMKL(int size, double* a, double* b, double* c, int thread_ID) {
	double factor = 2.0;
	for (unsigned int i = 0; i < size; i++)
	{
		*c = ((a[i] * b[i]) / (a[i] + b[i])) + *c;
	}
	*c = factor * *c;
}

int factorial(double n)
{
	return (n == 1.0 || n == 0.0) ? 1.0 : n * factorial(n - 1.0);
}

void calcCuMKL(int vDim, int& size, double* Cu) {

	double N = static_cast<double>(vDim);
	int vDim1 = 1;
	double fac, tmp_floor, tmp_pow;
	*Cu = 0;

	for (double i = 0; i < size; i++) {
		tmp_floor = i / N;
		vdFloor(vDim1, &tmp_floor, &tmp_floor);
		tmp_pow = N + i;
		vdPowx(vDim1, &tmp_pow, N, &tmp_pow);
		fac = factorial(tmp_floor);
		if (fac > 0)
			*Cu = 4.0 * tmp_pow / fac + *Cu;
		else {
			size = static_cast<int>(i);
			return;
		}
	}
}

void calcCvMKL(int vDim, int& size, double* Cv) {

	double N = static_cast<double>(vDim);
	int vDim1 = 1;
	double fac, tmp_pow;
	*Cv = 0;

	for (double i = 0; i < size; i++) {
		tmp_pow = N + i;
		vdPowx(vDim1, &tmp_pow, N, &tmp_pow);
		fac = factorial(i);
		if (fac > 0)
			*Cv = tmp_pow / fac + *Cv;
		else {
			size = static_cast<int>(i);
			return;
		}
	}
}

unsigned long long int epsBoundMKL(int m, int n, double Cu, double Cv, double eps) {

	int vDim1 = 1;
	double tmp_ceil, tmp_pow, pow, tmp_ln;

	tmp_ln = static_cast<double>(m + n + 1);
	vdLn(vDim1, &tmp_ln, &tmp_ln);
	pow = 2.0;
	tmp_pow = 1.0 + 2.0 * (Cu + Cv + 1.0) / eps;
	vdPowx(vDim1, &tmp_pow, pow, &tmp_pow);

	tmp_ceil = 8.0 * tmp_ln * tmp_pow;
	vdCeil(vDim1, &tmp_ceil, &tmp_ceil);
	return static_cast<unsigned long long int>(tmp_ceil);
}
#endif // MKL


#ifdef EXTERN
unsigned int getEpsRank(int m, int n, double* nlvm, double eps, double tol, int& no_conv_flag) {

	no_conv_flag = 0;
	bool last_iter = false;
	/* Local Variable Definitions */
	int minmn = std::min(m, n);
	unsigned int epsRank = minmn;
	std::vector<SV_ERR_> sv_err_vec_rmdr;
	std::vector<int> nsv_vec;
	SPLIT_BOUNDS bounds;

	/* Allocate Heap Memory */
	std::shared_ptr<double[]> U(new double[m * m], std::default_delete<double[]>());
	std::shared_ptr<double[]> s(new double[minmn], std::default_delete<double[]>());
	std::shared_ptr<double[]> VT(new double[n * n], std::default_delete<double[]>());
	std::shared_ptr<double[]> superb(new double[minmn - 1], std::default_delete<double[]>());
	std::shared_ptr<double[]> S(new double[m * n], std::default_delete<double[]>());
	std::shared_ptr<double[]> NLVM_CMPR(new double[m * n], std::default_delete<double[]>());

	/* Make a deep Copy of the NLVM-Matrix */
	std::memcpy(NLVM_CMPR.get(), nlvm, m * n * sizeof(double));

	/* Perform SVD*/
#ifdef MKL
	int status = svdMKL(m, n, nlvm, U.get(), s.get(), VT.get(), superb.get());
	gendiagm(m, n, s.get(), S.get());
#endif // MKL
	//arma svd as alternative, but not thread-safe!

	int n_threads;
#ifdef OMP
	n_threads = omp_get_max_threads(); // 8
#else
	n_threads = 8; // perform tasks of 8 threads in sequence
#endif // OMP

	/* Initial Domain Splitting */
	std::vector<SV_ERR_> sv_err_vec(n_threads);
	bounds = { 0,m }; // Domain Initialization
	splitDomain(sv_err_vec, sv_err_vec_rmdr, bounds, n_threads, m, INITIAL);
	int n_rmdr = sv_err_vec_rmdr.size();

	/* Initialize Loop Condition */
	for (int i = 0; i < n_threads; i++)
		sv_err_vec[i].err = 0;
	nsv_vec = extractNSV(sv_err_vec);

	/* Determine Epsilon-Rank of the NLVM-Matrix*/
	while (!std::equal(nsv_vec.begin() + 1, nsv_vec.end(), nsv_vec.begin()))
	{
		if (n_rmdr > 0)
			n_threads = n_rmdr;
#ifdef OMP	
		/* Determine Epsilon-Rank in Parallel */
#pragma omp parallel for default(none) shared(U,S,VT,sv_err_vec,n_threads,m,n,NLVM_CMPR)
		for (int i = 0; i < n_threads; i++) {

			/* Allocate Heap Memory */
			std::shared_ptr<double[]> TMP(new double[m * n], std::default_delete<double[]>());
			std::shared_ptr<double[]> NLVM_RECON(new double[m * n], std::default_delete<double[]>());

			/* Determine Error for specified Ranks */
			mreconMKL(m, n, sv_err_vec[i].nsv, U.get(), S.get(), VT.get(), NLVM_RECON.get(), TMP.get(), i);
			sv_err_vec[i].err = getmdiffMKL(m * n, NLVM_CMPR.get(), NLVM_RECON.get(), i);
		}
#else
		/* Determine Epsilon-Rank in Sequence */
		{
			/* Allocate Heap Memory */
			std::shared_ptr<double[]> TMP(new double[m * n], std::default_delete<double[]>());
			std::shared_ptr<double[]> NLVM_RECON(new double[m * n], std::default_delete<double[]>());

			/* Determine Error for specified Ranks */
			for (int i = 0; i < n_threads; i++) {
				mreconMKL(m, n, sv_err_vec[i].nsv, U.get(), S.get(), VT.get(), NLVM_RECON.get(), TMP.get(), i);
				sv_err_vec[i].err = getmdiffMKL(m * n, NLVM_CMPR.get(), NLVM_RECON.get(), i);
			}
		}
#endif // OMP

		/* Case: Error lays in (Epsilon - Epsilon*Tolerance, Epsilon + Epsilon*Tolerance) */
		if (checkIfInTolerance(sv_err_vec, eps, tol, epsRank, n_threads))
			return epsRank;

		/* Determine Refinement Domain */
		if (!last_iter)
		{
			bounds = checkForBitFlip(sv_err_vec, eps, m, n);
			if (bounds.upper==bounds.lower)
			{
				epsRank = bounds.upper;
				return epsRank;
			}
			if (bounds.upper - bounds.lower < n_threads)
			{
				last_iter = true;
				sv_err_vec_rmdr = sv_err_vec;
				sv_err_vec.clear();

				for (int i = bounds.lower; i <= bounds.upper; i++)
				{
					SV_ERR_ tmp{ i,0 };
					sv_err_vec.push_back(tmp);
				}
				n_rmdr = bounds.upper - bounds.lower + 1;
				continue;
			}
		}
		

		/* Case: No convergence. Closest Error to Epsilon */
		if (last_iter)
		{
			int tar_idx = handleNoConv(sv_err_vec.size(), sv_err_vec, eps, no_conv_flag);
			if (tar_idx > -1) {
				epsRank = sv_err_vec[tar_idx].nsv;
				return epsRank;
			}
		}

		/* Split Refinement Domain */
		if (splitDomain(sv_err_vec, sv_err_vec_rmdr, bounds, n_threads - 1, m, NOT_INITIAL))
		{
			if (sv_err_vec_rmdr.size() > 0)
			{
				n_rmdr = sv_err_vec_rmdr.size();
				sv_err_vec.clear();
				sv_err_vec = sv_err_vec_rmdr;
				last_iter = true;
			}
			else
				n_rmdr = sv_err_vec.size();
		}

		/* Update Loop Condition*/
		nsv_vec = extractNSV(sv_err_vec); // nsv_vec = \nu^m, here it is extracted from sv_err_vec
		if (nsv_vec.size() < 2)
		{
			no_conv_flag = 1;
			return(nsv_vec[0]);
		}
	}
}
#endif // EXTERN









#if 0
/* should be executed just once */
double err(n_threads);
std::fill(err.begin(), err.end(), eps + 1);

std::vector<unsigned int> epsRank(n_threads), iter(n_threads);
std::fill(epsRank.begin(), epsRank.end(), 0);
std::fill(iter.begin(), iter.end(), 0);

dvec sing;
dmat left, right;
std::vector<float> fractional_size(n_threads);
std::fill(fractional_size.begin(), fractional_size.end(), 2.0);

std::vector<float> tmp(n_threads);
std::vector<float> rate(n_threads);
std::fill(rate.begin(), rate.end(), 1.5);
float tol = 0.01; // make dependent on thread?
std::vector<dmat> approx(n_threads);

std::vector<int> nSingValsAdjusted(n_threads);
std::fill(nSingValsAdjusted.begin(), nSingValsAdjusted.end(), 0);
bool status = false;

status = arma::svd(left, sing, right, armaDmat);


std::vector<SV_ERR_> SV_ERR(n_threads);
std::vector<std::vector<SV_ERR_>> sv_err_vec(SV_ERR.size());

if (!status)
return 0; //dann versucht anderer thread nochmals svd ?

float mat_size = arma::size(armaDmat)[0];
std::fill(tmp.begin(), tmp.end(), mat_size / fractional_size[0]);
std::vector<int> nSingVals(n_threads);
std::fill(nSingVals.begin(), nSingVals.end(), int(tmp[0]));
/* until here */
// bis hier sollte es laufen

while (err[threadID] > eps)
{
	if (maxIter == iter[threadID])
		return 0;

	if (nSingVals[threadID] < 1)
		nSingVals[threadID] = 1;

	if (nSingVals[threadID] > mat_size - 1)
		nSingVals[threadID] = (int)mat_size;


	nSingValsAdjusted[threadID] = nSingVals[threadID];

	std::vector<bool> return_orig(n_threads);
	std::fill(return_orig.begin(), return_orig.end(), false);

	/* Composition */

	/* leftSingVecs unitary square matrix nxn, rightSingVecs unitary matrix mxm	*/
	std::vector<bool> compare1(n_threads);

	compare1[threadID] = nSingVals[threadID] < sing.n_rows;

	std::vector<bool> compare2(n_threads);

	compare2[threadID] = nSingVals[threadID] == sing.n_rows;

	if (compare1[threadID])
	{
		std::vector<dmat> u(n_threads);
		std::vector<dvec> svs(n_threads);
		std::vector<dmat> vt(n_threads);


		u[threadID] = left.cols(0, nSingVals[threadID] - 1);

		svs[threadID] = sing.rows(0, nSingVals[threadID] - 1);

		vt[threadID] = right.cols(0, nSingVals[threadID] - 1).t();
		approx[threadID] = u[threadID] * diagmat(svs[threadID]) * vt[threadID];
	}

	if (compare2[threadID])
	{

		return_orig[threadID] = true;

	}
	else // ?
	{
		dmat approx = zeros(sing.n_rows, sing.n_rows);
	}

	if (return_orig[threadID])
		err[threadID] = 0;
	else
		err[threadID] = matrixMaxNormP(armaDmat, approx[threadID], threadID);


	epsRank[threadID] = nSingVals[threadID];

	SV_ERR[threadID].err = err[threadID];
	SV_ERR[threadID].sv = nSingVals[threadID];
	sv_err_vec.push_back(SV_ERR);

	bool descent_found = false;
	sv_err_checkP(sv_err_vec[threadID], eps, descent_found, threadID);
	if (descent_found) {
		if (abs(sv_err_vec.end()[threadID][-2].err - eps) > abs(sv_err_vec.end()[threadID][-1].err - eps))
			epsRank[threadID] = sv_err_vec.end()[threadID][-1].sv;
		else
			epsRank[threadID] = sv_err_vec.end()[threadID][-2].sv;
		return epsRank[threadID];
	}

	if (abs(err[threadID] - eps) < tol * eps)
		return epsRank[threadID];

	if (err[threadID] < eps)
	{
		fractional_size[threadID] = ceil(fractional_size[threadID] * rate[threadID]);
		tmp[threadID] = mat_size / fractional_size[threadID];
		nSingVals[threadID] = nSingVals[threadID] - int(tmp[threadID]);
		err[threadID] = eps + 1;

		if (nSingVals[threadID] == nSingValsAdjusted[threadID]) {
			no_conv_flag = 1; // kritisch ? 
			return epsRank[threadID];
		}
		iter[threadID]++;
		//continue;
	}


	if (err[threadID] > eps)
	{
		fractional_size[threadID] = ceil(fractional_size[threadID] * rate[threadID]);
		tmp[threadID] = mat_size / fractional_size[threadID];
		nSingVals[threadID] = nSingVals[threadID] + int(tmp[threadID]);
		if (nSingVals[threadID] == nSingValsAdjusted[threadID]) {
			no_conv_flag = 1;
			return epsRank[threadID];
		}
		iter[threadID]++;
	}

}
return epsRank[threadID];
#endif 0