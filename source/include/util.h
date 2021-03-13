#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <vector>

#include <mac0.h>

enum FOLDER_ID
{
	SAMPLING,
	EX,
	RESULTS
};

struct SPLIT_BOUNDS
{
	int lower;
	int upper;
};

struct SV_ERR_
{
	int nsv;
	double err;
};

/* Visualization */
void printC1darray(double* c, int size, char const* name);
void printC1darray(double* c, int size);
void printC2darray(double* C, int m, int n, char const* name);
void printC2darray(double* C, int m, int n);

/* File Handling and Folder Navigation */
std::string get_current_dir();
int check_if_folder_exists(char const* path);
int check_if_file_exists(char const* fname);
int changeFolder(int folder_ID, char const* name, char const* trial_name, char const* Exn = "Ex1");
int manage_folder_structure(char const* path, char const* sample_name, char const* trial_name);
std::string create_file_name(char const* sample_name, char const* trial_name, char const* file_extension);
std::string create_file_name(char const* ex_name, char const* nlvm_name, char const* sample_name, char const* trial_name,
	char const* file_extension);
std::string create_file_name(char const* ex_name, char const* nlvm_name, char const* result_name, char const* sample_name,
	char const* trial_name, char const* file_extension);

/* Miscellaneous */
SPLIT_BOUNDS checkForBitFlip(std::vector<SV_ERR_> sv_err_vec, double eps, int m, int n);
	bool splitDomain(std::vector<SV_ERR_>& sv_err_vec, std::vector<SV_ERR_>& sv_err_vec_rmdr, SPLIT_BOUNDS bounds, int n_splits, int upper_bound);
bool checkIfInTolerance(std::vector<SV_ERR_> sv_err_vec, double eps, double tol, unsigned int& epsRank, int n_threads);
int handleNoConv(int n_rmdr, std::vector<SV_ERR_> sv_err_vec, double eps, int& no_conv_flag);
std::vector<int> extractNSV(std::vector<SV_ERR_> sv_err_vec);
#endif // !UTIL_H
