#ifndef API_MATLAB_WRAPPERS_MEX_HEADERS_H
#define API_MATLAB_WRAPPERS_MEX_HEADERS_H

// A file that wraps all MATLAB C++ API header includes.
// This is the place where all the warnings and issues with including
// MATLAB API header should be handled.

#include "arrus/common/compiler.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
COMPILER_DISABLE_MSVC_WARNINGS(4100 4189 4458 4702)

// Note: the below lines solves the issue with DLL_EXPORT_SYM macro redefinition
// This issue was observed for Matlab 2022 API + gcc >= 4.0.
#ifndef _WIN32
#   if __GNUC__ >= 4
#define DLL_EXPORT_SYM_TMP DLL_EXPORT_SYM
#undef DLL_EXPORT_SYM
#   endif
#endif
#include <mex.hpp>
#include <mexAdapter.hpp>
#ifndef _WIN32
#   if __GNUC__ >= 4
#undef DLL_EXPORT_SYM
#define DLL_EXPORT_SYM DLL_EXPORT_SYM_TMP
#undef DLL_EXPORT_SYM_TMP
#   endif
#endif

COMPILER_POP_DIAGNOSTIC_STATE

#endif//API_MATLAB_WRAPPERS_MEX_HEADERS_H
