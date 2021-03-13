#ifndef MATHFNS_H
#define MATHFNS_H

#include "mac0.h"

#ifdef MKL
struct GM2BDF_RESULT
{
	int status;
};

struct GETOMBYGEBRD_RESULT
{
	int status;
};

struct GETSVDFBIDIAGM_RESULT
{
	int status;
};
#endif // MKL

/* SVD */
#ifdef MKL
GM2BDF_RESULT gm2bdf(int a_mrows, int a_ncols, double* a, double* d, double* e, double* tauq, double* taup);
GETOMBYGEBRD_RESULT getombygebrd(char const id, int mrows, int ncols, int origm_mrows, int origm_ncols, double* a, double* tau);
GETSVDFBIDIAGM_RESULT getsvdfbidiagm(char const id, int b_size, int nRightSingVecs, int nLeftSingVecs,
	double* singVals_fromd, double* secDiag_frome, double* rightSingVecs, double* leftSingVecs, double* c);

int svdMKL(int m, int n, double* A, double* U, double* s, double* VT, double* superb);
void mreconMKL(int m, int n, int k, double* U, double* S, double* VT, double* A, double*TMP, int thread_ID);
#endif // MKL

/* NLVM */
#ifdef MKL
void linearKMKL(int size, double* a, double* b, double* c, int thread_ID);
void sedMKL(int size, double* a, double* b, double* c, int thread_ID);
void polynomiald2KMKL(int size, double* a, double* b, double* c, int thread_ID);
void polynomiald3KMKL(int size, double* a, double* b, double* c, int thread_ID);
void cosineKMKL(int size, double* a, double* b, double* c, int thread_ID);
void _01LossMKL(int size, double* a, double* b, double* c, int thread_ID);
void l1LossMKL(int size, double* a, double* b, double* c, int thread_ID);
void bsplineKMKL(int size, double* a, double* b, double* c, int thread_ID); // not used
#endif // MKL

/* No NLVM */
#ifdef MKL
void chisqrKMKL(int size, double* a, double* b, double* c, int thread_ID);
#endif // MKL

/* Miscellaneous */
#ifdef MKL
void mamupliMKL(int a_mrows, int a_ncols, int b_ncols, double* a, double* b, double* c);
void mamupliMKL(int a_mrows, int a_ncols, int b_ncols, int lda, double* a, int ldb, double* b, double* c, int thread_ID);

double getewlavMKL(int size, double* arr, int thread_ID);
double getmdiffMKL(int size, double* A, double* B, int thread_ID);
	unsigned int getEpsRank(int m, int n, double* nlvm, double eps, double tol, int& no_conv_flag);

int factorial(double n);
void bsplineMKL(int vsize, double* a, double deg, double* c, int thread_ID);
void calcCuMKL(int vDim, int& size, double* Cu);
void calcCvMKL(int vDim, int& size, double* Cv);
unsigned long long int epsBoundMKL(int m, int n, double Cu, double Cv, double eps); // logarithmic upper bound
#endif // MKL

#endif // !MATHFNS_H
