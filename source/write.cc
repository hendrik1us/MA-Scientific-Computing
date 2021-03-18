#include <fstream>
#include <sstream>
#include <iostream>

#include "read.h"
#include "write.h"
#include "util.h"
#include "matrixgen.h"
#include "mathfns.h"

#include <iomanip>

#ifdef ARMA
#include <armadillo>
#endif // ARMA

void writeSamples(char const* sample_name, char const* trial_name, int vDim, std::vector<SAMPLE_DIMS> sample_dims, sample_fn fn) {

	if (!manage_folder_structure(SAMPLE_PATH, sample_name, trial_name))
		return;

	std::ofstream myfile;
	std::string post_file_name1, post_file_name2, file_name1, file_name2;

	char* file_extension1 = "1.csv", * file_extension2 = "2.csv";
	post_file_name1 = create_file_name(sample_name, trial_name, file_extension1);
	post_file_name2 = create_file_name(sample_name, trial_name, file_extension2);

	int thread_ID = 0;

	for (int s = 0; s < sample_dims.size(); s++) {

		std::stringstream ss1, ss2;
		ss1 << s + 1 << post_file_name1;
		file_name1 = ss1.str();
		ss2 << s + 1 << post_file_name2;
		file_name2 = ss2.str();

#ifdef MKL
		std::unique_ptr<double[]> ns1 = std::make_unique<double[]>(sample_dims[s].m1 * vDim);
		std::unique_ptr<double[]> ns2 = std::make_unique<double[]>(sample_dims[s].m2 * vDim);

		nRandomVectorsCreateMKL(sample_dims[s].m1, vDim, ns1.get(), fn, thread_ID);
		nRandomVectorsCreateMKL(sample_dims[s].m2, vDim, ns2.get(), fn, thread_ID);
#endif // MKL
#ifdef ARMA
		dmat ns1 = nRandomVectorsCreate(nsamples_ vDim, unifSampleFromDSphere);
		dmat ns2 = nRandomVectorsCreate(nsamples_, vDim, unifSampleFromDSphere);
#endif // ARMA

		int prec = std::numeric_limits<double>::digits10 + 2;
		/* Write First Dataset */
		if (!myfile.is_open()) {
			myfile.open(file_name1);
		}
		std::cout << "Start writing " << file_name1 << std::endl;

		for (int i = 0; i < sample_dims[s].m1; i++)
		{
			for (int j = 0; j < vDim; j++)
			{
				if (j < vDim - 1) {
#ifdef MKL
					myfile << std::fixed << std::setprecision(prec) << ns1.get()[i * vDim + j] << ",";
#endif // MKL
#ifdef ARMA
					myfile << std::fixed << std::setprecision(prec) << ns1(i, j) << ",";
#endif // ARMA
				}
				else {
#ifdef MKL
					myfile << std::fixed << std::setprecision(prec) << ns1.get()[i * vDim + j] << "\n";
#endif // MKL
#ifdef ARMA
					myfile << std::fixed << std::setprecision(prec) << ns1(i, j) << "\n";
#endif // ARMA
				}
			}
		}
		myfile.close();
		std::cout << "Finished writing " << file_name1 << std::endl;

		/* Write Second Dataset*/
		if (!myfile.is_open()) {
			myfile.open(file_name2);
		}
		std::cout << "Start writing " << file_name1 << std::endl;

		for (int i = 0; i < sample_dims[s].m2; i++)
		{
			for (int j = 0; j < vDim; j++)
			{
				if (j < vDim - 1) {
#ifdef MKL
					myfile << std::fixed << std::setprecision(prec) << ns2.get()[i * vDim + j] << ",";
#endif // MKL
#ifdef ARMA
					myfile << std::fixed << std::setprecision(prec) << ns2(i, j) << ",";
#endif // ARMA
				}
				else {
#ifdef MKL
					myfile << std::fixed << std::setprecision(prec) << ns2.get()[i * vDim + j] << "\n";
#endif // MKL
#ifdef ARMA
					myfile << std::fixed << std::setprecision(prec) << ns2(i, j) << "\n";
#endif // ARMA
				}
			}
		}
		myfile.close();
		std::cout << "Finished writing " << file_name2 << "\n" << std::endl;
	}
}

void writeSamples(char const* sample_name, char const* trial_name, int vDim, std::vector<SAMPLE_DIMS> sample_dims, sample_fn fn, double scale) {

	if (!manage_folder_structure(SAMPLE_PATH, sample_name, trial_name))
		return;

	std::ofstream myfile;
	std::string post_file_name1, post_file_name2, file_name1, file_name2;

	char* file_extension1 = "1.csv", * file_extension2 = "2.csv";
	post_file_name1 = create_file_name(sample_name, trial_name, file_extension1);
	post_file_name2 = create_file_name(sample_name, trial_name, file_extension2);

	int thread_ID = 0;

	for (int s = 0; s < sample_dims.size(); s++) {

		std::stringstream ss1, ss2;
		ss1 << s + 1 << post_file_name1;
		file_name1 = ss1.str();
		ss2 << s + 1 << post_file_name2;
		file_name2 = ss2.str();

#ifdef MKL
		std::unique_ptr<double[]> ns1 = std::make_unique<double[]>(sample_dims[s].m1 * vDim);
		std::unique_ptr<double[]> ns2 = std::make_unique<double[]>(sample_dims[s].m2 * vDim);

		nRandomVectorsCreateMKL(sample_dims[s].m1, vDim, ns1.get(), fn, thread_ID);
		nRandomVectorsCreateMKL(sample_dims[s].m2, vDim, ns2.get(), fn, thread_ID);
#endif // MKL
#ifdef ARMA
		dmat ns1 = nRandomVectorsCreate(nsamples_ vDim, unifSampleFromDSphere);
		dmat ns2 = nRandomVectorsCreate(nsamples_, vDim, unifSampleFromDSphere);
#endif // ARMA

		int prec = std::numeric_limits<double>::digits10 + 2;
		/* Write First Dataset */
		if (!myfile.is_open()) {
			myfile.open(file_name1);
		}
		std::cout << "Start writing " << file_name1 << std::endl;

		for (int i = 0; i < sample_dims[s].m1; i++)
		{
			for (int j = 0; j < vDim; j++)
			{
				if (j < vDim - 1) {
#ifdef MKL
					myfile << std::fixed << std::setprecision(prec) << ns1.get()[i * vDim + j] << ",";
#endif // MKL
#ifdef ARMA
					myfile << std::fixed << std::setprecision(prec) << ns1(i, j) << ",";
#endif // ARMA
				}
				else {
#ifdef MKL
					myfile << std::fixed << std::setprecision(prec) << ns1.get()[i * vDim + j] << "\n";
#endif // MKL
#ifdef ARMA
					myfile << std::fixed << std::setprecision(prec) << ns1(i, j) << "\n";
#endif // ARMA
				}
			}
		}
		myfile.close();
		std::cout << "Finished writing " << file_name1 << std::endl;

		/* Write Second Dataset*/
		if (!myfile.is_open()) {
			myfile.open(file_name2);
		}
		std::cout << "Start writing " << file_name1 << std::endl;

		for (int i = 0; i < sample_dims[s].m2; i++)
		{
			for (int j = 0; j < vDim; j++)
			{
				if (j < vDim - 1) {
#ifdef MKL
					myfile << std::fixed << std::setprecision(prec) << ns2.get()[i * vDim + j] << ",";
#endif // MKL
#ifdef ARMA
					myfile << std::fixed << std::setprecision(prec) << ns2(i, j) << ",";
#endif // ARMA
				}
				else {
#ifdef MKL
					myfile << std::fixed << std::setprecision(prec) << ns2.get()[i * vDim + j] << "\n";
#endif // MKL
#ifdef ARMA
					myfile << std::fixed << std::setprecision(prec) << ns2(i, j) << "\n";
#endif // ARMA
				}
			}
		}
		myfile.close();
		std::cout << "Finished writing " << file_name2 << "\n" << std::endl;
	}
}


void writeNLVMs(char const* nlvm_name, char const* Exn, char const* sample_name, char const* trial_name, std::vector<NLVM_DIMS> nlvm_dims,
	nlvmf fn) {

	std::ofstream myfile;
	std::string post_rfile_name1, rfile_name1, post_rfile_name2, rfile_name2, post_wfile_name, wfile_name;

	char* rfile_extension1 = "1.csv", * rfile_extension2 = "2.csv", * wfile_extension = ".csv";

	int thread_ID = 0;

	for (int s = 0; s < nlvm_dims.size(); s++) {

		post_rfile_name1 = create_file_name(sample_name, trial_name, rfile_extension1);
		post_rfile_name2 = create_file_name(sample_name, trial_name, rfile_extension2);

		std::stringstream ss1, ss2;
		ss1 << s + 1 << post_rfile_name1;
		rfile_name1 = ss1.str();
		ss2 << s + 1 << post_rfile_name2;
		rfile_name2 = ss2.str();

#ifdef MKL
		std::unique_ptr<double[]> nlvm_mat = std::make_unique<double[]>(nlvm_dims[s].m * nlvm_dims[s].n);
#ifdef EXTERN
		nlvm(nlvm_dims[s].m, nlvm_dims[s].n, nlvm_mat.get(), fn, sample_name, trial_name, rfile_name1.c_str(), rfile_name2.c_str(), thread_ID);
#endif // EXTERN
#endif // MKL

#ifdef ARMA
#endif // ARMA

		post_wfile_name = create_file_name(Exn, nlvm_name, sample_name, trial_name, wfile_extension);

		std::stringstream ss;
		ss << s + 1 << post_wfile_name;
		wfile_name = ss.str();

		int prec = std::numeric_limits<double>::digits10 + 2;
		if (!manage_folder_structure(EXAMPLE_PATH, Exn, trial_name))
			return;

		/* Write NLVM Matrix */
		if (!myfile.is_open()) {
			myfile.open(wfile_name);
		}
		std::cout << "Start writing " << wfile_name << std::endl;

		for (int i = 0; i < nlvm_dims[s].m; i++)
		{
			for (int j = 0; j < nlvm_dims[s].n; j++)
			{
				if (j < nlvm_dims[s].n - 1) {
#ifdef MKL
					myfile << std::fixed << std::setprecision(prec) << nlvm_mat.get()[i * nlvm_dims[s].n + j] << ",";
#endif // MKL
#ifdef ARMA
					myfile << std::fixed << std::setprecision(prec) << nlvm_mat(i, j) << ",";
#endif // ARMA
				}
				else {
#ifdef MKL
					myfile << std::fixed << std::setprecision(prec) << nlvm_mat.get()[i * nlvm_dims[s].n + j] << "\n";
#endif // MKL
#ifdef ARMA
					myfile << std::fixed << std::setprecision(prec) << nlvm_mat(i, j) << "\n";
#endif // ARMA
				}
			}
		}
		myfile.close();
		std::cout << "Finished writing " << wfile_name << std::endl;
	}
}

void writeResults(char const* nlvm_name, char const* Exn, char const* result_name, char const* sample_name, char const* trial_name,
	std::vector<NLVM_DIMS> nlvm_dims) {

	std::ofstream myfile;
	std::string rfile_name, post_rfile_name, wfile_name;

	char* file_extension = ".csv";

	for (int s = 0; s < nlvm_dims.size(); s++)
	{
		std::unique_ptr<double[]> NLVM = std::make_unique<double[]>(nlvm_dims[s].m * nlvm_dims[s].n);
		std::unique_ptr<double[]> NLVM_CMPR = std::make_unique<double[]>(nlvm_dims[s].m * nlvm_dims[s].n);

		post_rfile_name = create_file_name(Exn, nlvm_name, sample_name, trial_name, file_extension);

		std::stringstream ss;
		ss << s + 1 << post_rfile_name;
		rfile_name = ss.str();

		if (changeFolder(EX, sample_name, trial_name,Exn) < 0)
			return;
		if (!check_if_file_exists(rfile_name.c_str()))
			return;

		std::string dummy =  get_current_dir();


		int thread_ID = 0;
		std::cout << "Start reading " << rfile_name << std::endl;
		std::vector<std::vector<double>> nlvm_vecvec{ readCSV(rfile_name.c_str(),thread_ID) };
		dVecVecToCArr(nlvm_vecvec, NLVM.get(), thread_ID);
		nlvm_vecvec.swap(std::vector<std::vector<double>>()); // ?
		std::cout << "Reading complete" << std::endl;

		//dmat armaX0 = vecVecToArmaMat(X);

		if (s < 1)
		{
			wfile_name = create_file_name(Exn, nlvm_name, result_name, sample_name, trial_name, file_extension);

			if (!manage_folder_structure(RESULTS_PATH, Exn, sample_name))
				return;
			if (check_if_file_exists(wfile_name.c_str()))
				return;
		}

		std::cout << "Start analyzing and writing " << std::endl;
		if (!myfile.is_open()) {
			myfile.open(wfile_name);
			myfile << "nrows,ncols,eps,epsRank,flag\n";
		}

		unsigned int epsRank;
		int no_conv_flag = 0;
		double tol = 0.001;
		std::vector<float> epsVec = { 0.2, 0.1, 0.07, 0.03, 0.01, 0.001 }; //SED
		epsVec = { 0.1, 0.07, 0.03, 0.01, 0.005, 0.001 }; //LINEARK
		epsVec = { 0.03, 0.01, 0.007, 0.005,0.0025, 0.001 }; //POLYNOMIALK d=2
		epsVec = { 0.25, 0.15, 0.05,0.1, 0.01,0.001 }; //COSINEK
		epsVec = { 0.1, 0.06, 0.03, 0.02, 0.01, 0.005}; ; //POLYNOMIALK d=3
		epsVec = {150000,100000,50000,25000, 10000, 1000}; //CHISQRK
		epsVec = { 2, 1, 0.5, 0.3, 0.17, 0.03 }; //L1LOSSMKL
		//epsVec = { 1.3, 1.0, 0.7, 0.4, 0.2, 0.05 }; // 01LOSSMKL 

		for (auto eps : epsVec)
		{
			std::memcpy(NLVM_CMPR.get(), NLVM.get(), nlvm_dims[s].m * nlvm_dims[s].n * sizeof(double));
			epsRank = getEpsRank(nlvm_dims[s].m, nlvm_dims[s].n, NLVM_CMPR.get(), eps, tol, no_conv_flag);
			myfile << nlvm_dims[s].m << "," << nlvm_dims[s].n << "," << eps << "," << epsRank << "," << no_conv_flag << "\n";
		}
	}
	myfile.close();
}