#ifndef MAC0_H
#define MAC0_H

#define OMP 
#define MKL
#define MKL_TEST 0

#define EXTERN
#define EXTERN_REPO

#ifdef EXTERN_REPO
constexpr char const* SAMPLE_PATH = "Q:\\Projects\\MA\\Sampling\\";
constexpr char const* EXAMPLE_PATH = "Q:\\Projects\\MA\\Ex\\";
constexpr char const* RESULTS_PATH = "Q:\\Projects\\MA\\Results\\";
#else 
constexpr char const* SAMPLE_PATH = "C:\\Users\\h_dei\\Documents\\Master TU Berlin\\Module\\Masterarbeit\\Data\\Sampling\\";
constexpr char const* EXAMPLE_PATH = "C:\\Users\\h_dei\\Documents\\Master TU Berlin\\Module\\Masterarbeit\\Data\\Ex\\";
constexpr char const* RESULTS_PATH = "C:\\Users\\h_dei\\Documents\\Master TU Berlin\\Module\\Masterarbeit\\Data\\Results\\";
#endif // EXTERN_REPO


//#define TBB //alternative to OMP

//#define DEBUG

#ifdef DEBUG

#ifndef TBB
//#undef _DEBUG
#define _CRTDBG_MAP_ALLOC
#endif // !TBB


#endif // DEBUG


#endif // !MAC0_H

