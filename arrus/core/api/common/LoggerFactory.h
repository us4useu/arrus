#ifndef ARRUS_CORE_API_COMMON_LOGGERFACTORY_H
#define ARRUS_CORE_API_COMMON_LOGGERFACTORY_H

#include <utility>
#include <vector>

#include "Logger.h"

namespace arrus {

class LoggerFactory {
public:
    virtual Logger::Handle getLogger() = 0;

    virtual Logger::Handle
    getLogger(const std::vector<arrus::Logger::Attribute> &attributes) = 0;

    virtual ~LoggerFactory() = default;
};

}

#endif //ARRUS_CORE_API_COMMON_LOGGERFACTORY_H
