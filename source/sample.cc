#include <iostream>
#include <vector>

#include "mac0.h"

#ifdef MKL
#include <mkl.h>
#include <algorithm>
#include <iterator>
#endif // MKL

#include "sample.h"
#include "util.h"

#ifdef OMP
#include <omp.h>
#endif // OMP


#ifdef ARMA
drowvec unifSampleFromDBall(unsigned int dim) {
	/*std::uniform_real_distribution<> dis(0, 1.0);
	drowvec vector(dim);
	double r = pow(dis(gen), 1.0 / dim);
	vector.randn();
	vector = arma::normalise(vector, 2);
	vector = r * vector;
	return vector;*/
	return { 0 };
}

drowvec unifSampleFromDSphere(unsigned int dim) {
	drowvec vector = arma::randn(1, dim);
	vector = arma::normalise(vector, 2);
	return vector;
}

dmat nRandomVectorsCreate(unsigned int nVecs, unsigned int vecDim, sample_fn fn) {
	dmat base(nVecs, vecDim);
	for (unsigned int i = 0; i < nVecs; i++)
	{
		base.row(i) = fn(vecDim);
	}
	return base;
}
#endif // ARMA

#ifdef MKL
void unifSampleFromNSphereMKL(int size, double* s, int thread_ID) {

	double mean = 0.0;
	double var = 1.0;

	/* Buffer for random numbers */
	VSLStreamStatePtr stream;

	/* Initializing */
	vslNewStream(&stream, /*VSL_BRNG_MT19937*/ VSL_BRNG_NONDETERM, 0 /*777*/);

	/* Generating */
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, size, s, mean, var);

	/* Deleting the stream */
	vslDeleteStream(&stream);

	/* Calculating the euclidean norm of random vector r */
	int inc = 1;
	double norm = cblas_dnrm2(size, s, inc);
	double scale = 1.0 / norm;

	/* Scale the random vector r */
	cblas_dscal(size, scale, s, inc);
}

void unifSampleFromN2SphereMKL(int size, double* s, int thread_ID) {

	double mean = 0.0;
	double var = 1.0;

	/* Buffer for random numbers */
	VSLStreamStatePtr stream;

	/* Initializing */
	vslNewStream(&stream, /*VSL_BRNG_MT19937*/ VSL_BRNG_NONDETERM, 0 /*777*/);

	/* Generating */
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, size, s, mean, var);

	/* Deleting the stream */
	vslDeleteStream(&stream);

	/* Calculating the euclidean norm of random vector r */
	int inc = 1;
	double norm = cblas_dnrm2(size, s, inc);
	double scale = 2.0 / norm;

	/* Scale the random vector r */
	cblas_dscal(size, scale, s, inc);
}

void unifSampleFromNBallMKL(int size, double* s, int thread_ID) {

	/* Create random vector in unit sphere */
	unifSampleFromNSphereMKL(size, s, thread_ID);

	/* Scale random vector s.t. it points to an abitrary point inside the N-ball */
	double pow = 1.0 / static_cast<double>(size);
	MKL_INT seed[4] = { 1,2,3,4 };
	double unifrn = dlaran(seed);
	double scale = 0;
	vdPowx(1, &unifrn, pow, &scale);
	int inc = 1;
	cblas_dscal(size, scale, s, inc);
}

void unifSampleFromN2BallMKL(int size, double* s, int thread_ID) {

	/* Create random vector in unit sphere */
	unifSampleFromN2SphereMKL(size, s, thread_ID);

	/* Scale random vector s.t. it points to an abitrary point inside the N2-ball */
	double pow = 1.0 / static_cast<double>(size);
	MKL_INT seed[4] = { 1,2,3,4 };
	double unifrn = dlaran(seed);
	double scale = 0;
	vdPowx(1, &unifrn, pow, &scale);
	int inc = 1;
	cblas_dscal(size, scale, s, inc);
}

void nRandomVectorsCreateMKL(int m, int n, double* ns, sample_fn fn, int thread_ID) {

	std::vector<std::unique_ptr<double[]>> shared_tmp;
	for (int s = 0; s < m; s++)
	{
		shared_tmp.push_back(std::move(std::make_unique<double[]>(n)));
	}

#ifdef OMP
#pragma omp parallel for default(none) shared(n,m,ns,shared_tmp,fn,thread_ID)
#endif // OMP
	for (int i = 0; i < m; i++)
	{
		fn(n, shared_tmp[i].get(), i); // Do sampling
		for (int j = 0; j < n; j++)
		{
#ifdef DEBUG
			printf("Hello from thread %i. I work on index pair (%i,%i)\n",
				omp_get_thread_num(), i, j);
#endif // DEBUG	
			ns[i * n + j] = shared_tmp[i].get()[j]; // Assign to "matrix"
		}
	}
}
#endif // MKL

