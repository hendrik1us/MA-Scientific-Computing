#ifndef MATRIXGEN_H
#define MATRIXGEN_H

#include "mac0.h"

#include "sample.h"

typedef void(*nlvmf)(int size, double* a, double* b, double* c, int id);

void gendiagm(int m, int n, double* d, double* diagm);

#ifdef INTERN
void nlvm(int m, int n, double* nlvm, nlvmf nlvm_fn, int vDim, sample_fn s_fn, int thread_ID);
#endif // INTERN

#ifdef EXTERN
void nlvm(int m, int n, double* nlvm, nlvmf nlvm_fn, char const* sample_name, char const* trial_name, char const* id1, char const* id2, int thread_ID);
#endif // EXTERN

#endif // !MATRIXGEN_H
