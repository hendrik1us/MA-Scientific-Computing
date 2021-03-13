#ifndef SAMPLE_H
#define SAMPLE_H

#include <vector>

#include "mac0.h"

#ifdef MKL
#include <memory>
typedef void(*sample_fn)(int size, double* s, int id);
#endif // MKL

#ifdef ARMA
#include <armadillo>
#endif // ARMA

#ifdef TBB
#include <tbb/tbb.h>
#endif // !TBB

#ifdef ARMA
using namespace arma; //besser nicht verwenden
typedef drowvec(*sample_fn)(unsigned int);
typedef double(*f)(drowvec, drowvec);

drowvec unifSampleFromDBall(unsigned int dim);
drowvec unifSampleFromDSphere(unsigned int dim);
dmat nRandomVectorsCreate(unsigned int nVecs, unsigned int vecDim, sample_fn fn);
#endif // ARMA

#ifdef MKL
void unifSampleFromNSphereMKL(int size, double* s, int thread_ID);
void unifSampleFromN2SphereMKL(int size, double* s, int thread_ID);
void unifSampleFromNBallMKL(int size, double* s, int thread_ID);
void unifSampleFromN2BallMKL(int size, double* s, int thread_ID);
void nRandomVectorsCreateMKL(int m, int n, double* ns, sample_fn fn, int thread_ID);
#endif // MKL

#endif // !SAMPLE_H
