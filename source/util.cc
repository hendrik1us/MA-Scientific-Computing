#include <stdio.h>
#include <direct.h>
#include <iostream>
#include <vector>
#include <algorithm>

#include <mac0.h>

#include "util.h"

/* Visualization */

void printC1darray(double* c, int size, char const* name) {
	for (unsigned int i = 0; i < size; i++) {
		if (name[0] != 0)
		{
			printf(name);
			printf("_%d = %f\n", i, c[i]);
		}
		else
		{
			printf("%f ", c[i]);
		}
	}
	printf("\n");
}

void printC1darray(double* c, int size) {
	char null[1];
	null[0] = 0;
	printC1darray(c, size, null);
}

void printC2darray(double* C, int m, int n, char const* name) {
	unsigned int start_idx;
	for (unsigned int i = 0; i < m; i++) {
		start_idx = i * n;
		for (unsigned int j = 0; j < n; j++) {
			if (name[0] != 0)
			{
				printf(name);
				printf("_%d%d = %f ", i, j, C[start_idx + j]);
			}
			else
			{
				printf("%f ", C[start_idx + j]);
			}
		}
		printf("\n");
	}
	printf("\n");
}

void printC2darray(double* C, int m, int n) {
	char null[1];
	null[0] = 0;
	printC2darray(C, m, n, null);
}


/* File Handling and Folder Navigation */

std::string get_current_dir() {

	char buff[FILENAME_MAX];
	_getcwd(buff, FILENAME_MAX);
	std::string current_working_dir(buff);
	return current_working_dir;
}

int check_if_folder_exists(char const* path) {
	struct stat info;
	int state = 0;
	if (stat(path, &info) != 0)
		printf("cannot access %s\n", path);
	else if (info.st_mode & S_IFDIR) {
		printf("%s is a directory\n", path);
		state = 1;
	}
	else
		printf("%s is no directory\n", path);
	return state;
}

int manage_folder_structure(char const* path, char const* sample_name, char const* trial_name) {
	int state = 1;
	std::string buffer(path);

	if (!check_if_folder_exists(buffer.c_str()))
		mkdir(buffer.c_str());
	buffer.append(sample_name);

	if (!check_if_folder_exists(buffer.c_str()))
		mkdir(buffer.c_str());

	char const* bsl = "\\";

	if (trial_name[0] != '\0')
	{
		buffer.append(bsl);
		buffer.append(trial_name);
	}

	if (!check_if_folder_exists(buffer.c_str()))
		mkdir(buffer.c_str());

	chdir(buffer.c_str());
	std::string dir = get_current_dir();
	if (buffer != dir)
		state = 0;

	return state;
}

std::string create_file_name(char const* sample_name, char const* trial_name, char const* file_extension) {
	char const* prefix = "MA_Dat_";
	char const* underscore = "_";

	std::string buffer(prefix);
	buffer.append(sample_name);
	buffer.append(underscore);
	buffer.append(trial_name);
	if (file_extension[0] != 0)
		buffer.append(underscore);
	buffer.append(file_extension);
	return buffer;
}

std::string create_file_name(char const* ex_name, char const* nlvm_name, char const* sample_name, char const* trial_name, char const* file_extension) {
	char const* prefix = "MA_Dat_";
	char const* underscore = "_";

	std::string buffer1(create_file_name(sample_name, trial_name, file_extension));
	buffer1.erase(0, strlen(prefix));
	std::string buffer2(prefix);
	buffer2.append(ex_name);
	buffer2.append(underscore);
	buffer2.append(nlvm_name);
	buffer2.append(underscore);
	buffer1.erase(buffer1.end() - strlen(file_extension) - 1);
	return buffer2.append(buffer1);
}

std::string create_file_name(char const* ex_name, char const* nlvm_name, char const* result_name, char const* sample_name, char const* trial_name, char const* file_extension) {
	char const* prefix = "MA_Res_";
	char const* underscore = "_";

	std::string buffer1(create_file_name(sample_name, trial_name, file_extension));
	buffer1.erase(0, strlen(prefix));
	std::string buffer2(prefix);
	buffer2.append(ex_name);
	buffer2.append(underscore);
	buffer2.append(nlvm_name);
	buffer2.append(underscore);
	buffer2.append(result_name);
	buffer2.append(underscore);
	buffer1.erase(buffer1.end() - strlen(file_extension) - 1);
	return buffer2.append(buffer1);
}

int changeFolder(int folder_ID, char const* name, char const* trial_name, char const* Exn) {
	char const* bsl = "\\";
	std::string buffer;
	if (folder_ID == SAMPLING)
		buffer = SAMPLE_PATH;
	else if (folder_ID == EX) {
		buffer = EXAMPLE_PATH;
		buffer.append(Exn);
	}
	else
		buffer = RESULTS_PATH;
	if (folder_ID != EX)
		buffer.append(name);

	if (folder_ID != RESULTS)
	{
		buffer.append(bsl);
		buffer.append(trial_name);
	}
	if (!check_if_folder_exists(buffer.c_str()))
		return -1;
	return chdir(buffer.c_str());
}

int check_if_file_exists(char const* fname) {
	struct stat buffer;
	return (stat(fname, &buffer) == 0);
}

/* Miscellaneous */

SPLIT_BOUNDS checkForBitFlip(std::vector<SV_ERR_> sv_err_vec, double eps, int m, int n) {
	std::vector<bool> less_eps(sv_err_vec.size());
	int count_bit_flips = 0;
	SPLIT_BOUNDS bounds = { 0,0 };
	for (int j = 0; j < sv_err_vec.size(); j++)
	{
		less_eps[j] = sv_err_vec[j].err < eps;
		if (j > 0)
		{
			if (less_eps[j - 1] != less_eps[j])
			{
				if (sv_err_vec[j].nsv - 1 - (sv_err_vec[j - 1].nsv + 1) <=0)
				{
					bounds = { sv_err_vec[j].nsv - 1 ,sv_err_vec[j].nsv - 1 };
				}
				else 
				{
					bounds = { sv_err_vec[j - 1].nsv + 1, sv_err_vec[j].nsv - 1 };
				}
				count_bit_flips = count_bit_flips + 1;
			}
		}
	}
	if (count_bit_flips > 0)
	{
		return bounds;
	}

	if (std::equal(less_eps.begin() + 1, less_eps.end(), less_eps.begin()))
	{
		int size = sv_err_vec[1].nsv - sv_err_vec[0].nsv -2;
		if (!less_eps[0])
		{
			bounds = { sv_err_vec[sv_err_vec.size() - 1].nsv + 1, std::min(sv_err_vec[sv_err_vec.size() - 1].nsv +1 + size,m-1) };
		}
		else
		{
			bounds = { std::max(sv_err_vec[0].nsv-1 - size,1), sv_err_vec[0].nsv };
		}
		return bounds;
	}
}

bool splitDomain(std::vector<SV_ERR_>& sv_err_vec, std::vector<SV_ERR_>& sv_err_vec_rmdr, SPLIT_BOUNDS bounds, int n_splits, int upper_bound, int CASE) {
	int domain_size = bounds.upper - bounds.lower;
	if (CASE == INITIAL)
		n_splits = n_splits - 1;

	double size_splitted = static_cast<double>(domain_size) / static_cast<double>(n_splits+1);
	bool rmdr_filled = false;
	double err = 0.0;
	int c = 0;

	if (size_splitted <= 1)
	{
		for (int i = 0; i < domain_size; i++)
			sv_err_vec_rmdr.push_back({ bounds.lower + i, err });
		rmdr_filled = true;
	}
	else
	{
		for (int i = 0; i < n_splits+1; i++) {
			if (CASE==INITIAL)
			{
				sv_err_vec[i].nsv = bounds.lower + (i + 1)* std::floor(size_splitted);
			}
			else
			{
				if (i == n_splits) {
					sv_err_vec[i].nsv = bounds.upper;
				}
				else
				{
					sv_err_vec[i].nsv = bounds.lower + i * std::floor(size_splitted);
				}
			}
			if (sv_err_vec[i].nsv > upper_bound) {
				sv_err_vec.erase(sv_err_vec.begin() + i, sv_err_vec.begin() + sv_err_vec.size());
				rmdr_filled = true;
				break;
			}
		}
	}
	return rmdr_filled;
}

bool checkIfInTolerance(std::vector<SV_ERR_> sv_err_vec, double eps, double tol, unsigned int& epsRank, int n_threads) {
	bool inTol = false;
	for (int i = 0; i < n_threads; i++)
		if ((sv_err_vec[i].err <= eps + tol * eps) && (sv_err_vec[i].err >= eps - tol * eps)) {
			epsRank = sv_err_vec[i].nsv;
			inTol = true;
		}
	return inTol;
}

int handleNoConv(int n_rmdr, std::vector<SV_ERR_> sv_err_vec, double eps, int& no_conv_flag) {
	int tar_idx = -1;
	no_conv_flag = 0;
	if (n_rmdr > 0)
	{
		std::vector<double> eps_cmpr_vec(n_rmdr, eps);
		for (int i = 0; i < n_rmdr; i++)
			eps_cmpr_vec[i] = std::abs(eps_cmpr_vec[i] - sv_err_vec[i].err);
		auto min_it = std::min_element(eps_cmpr_vec.begin(), eps_cmpr_vec.end());
		tar_idx = min_it - eps_cmpr_vec.begin();
		no_conv_flag = 1;
	}
	return tar_idx;
}

std::vector<int> extractNSV(std::vector<SV_ERR_> sv_err_vec) {
	std::vector<int> nsv_vec;
	for (int i = 0; i < sv_err_vec.size(); i++)
		nsv_vec.push_back(sv_err_vec[i].nsv);
	return nsv_vec;
}
