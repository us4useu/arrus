#ifndef ARRUS_CORE_COMMON_OS_H
#define ARRUS_CORE_COMMON_OS_H

#include <string_view>

namespace arrus {

#if defined(_WIN32)
constexpr std::string_view OS_NAME = "Windows";
#elif defined(__APPLE__) && defined(__MACH__)
constexpr std::string_view OS_NAME = "MacOS";
#elif defined(__linux__)
constexpr std::string_view OS_NAME = "Linux";
#else
constexpr std::string_view OS_NAME = "Unrecognized OS";
#endif

}

#endif//ARRUS_CORE_COMMON_OS_H
