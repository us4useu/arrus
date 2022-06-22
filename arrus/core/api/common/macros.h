#ifndef ARRUS_CORE_API_COMMON_MACROS_H
#define ARRUS_CORE_API_COMMON_MACROS_H

// Platform agnostic __declspec handling.
// Use ARRUS_CPP_EXPORT to mark specific structures/functions/etc.
// that should be accessible by the dll user.
// For systems other than windows, ARRUS_CPP is just an empty
// macro.
#define ARRUS_PATH_KEY "ARRUS_PATH"
#if defined(_WIN32) && !defined(ARRUS_CORE_UNIT_TESTS)

#if defined(ARRUS_CPP_API_BUILD_STAGE)
#define ARRUS_CPP_EXPORT __declspec(dllexport)
#else
#define ARRUS_CPP_EXPORT __declspec(dllimport)
#endif

#else
#define ARRUS_CPP_EXPORT __attribute__((visibility("default")))
#endif

#endif //ARRUS_CORE_API_COMMON_MACROS_H
