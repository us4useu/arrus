#ifndef ARRUS_CORE_API_COMMON_MACROS_H
#define ARRUS_CORE_API_COMMON_MACROS_H

#define ARRUS_PATH_KEY "ARRUS_PATH"

#if defined(_WIN32)

#if defined(ARRUS_CPP_API_BUILD_STAGE)
#define ARRUS_CPP_EXPORT __declspec(dllexport)
#else
#define ARRUS_CPP_EXPORT __declspec(dllimport)
#endif

#else
#define ARRUS_CPP_EXPORT
#endif

#endif //ARRUS_CORE_API_COMMON_MACROS_H
