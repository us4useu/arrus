#ifndef ARRUS_CORE_DEVICES_UTILS_H
#define ARRUS_CORE_DEVICES_UTILS_H

#include <string>
#include <range/v3/all.hpp>
#include "arrus/core/common/format.h"
#include "arrus/core/api/common/exceptions.h"

namespace arrus {

// Device path
constexpr char PATH_DELIMITER = '/';

std::pair<std::string, std::string> getPathRoot(const std::string &path) {
    if(path.empty() || path[0] != PATH_DELIMITER) {
        throw IllegalArgumentException(
                ::arrus::format("Invalid path '{}', should start with '{}'",
                                path, PATH_DELIMITER));
    }

    std::string relPath = path.substr(1, path.size() - 1);
    if(relPath.empty()) {
        throw IllegalArgumentException(
                arrus::format("Path should refer to some object "
                              "(got: '{}')", path));
    }

    size_t firstElementEnd = relPath.find(PATH_DELIMITER);
    if(firstElementEnd == std::string::npos) {
        return {relPath, ""};
    } else {
        return {
            relPath.substr(0, firstElementEnd),
            relPath.substr(firstElementEnd, relPath.size()-firstElementEnd)
        };
    }
}

}

#endif //ARRUS_CORE_DEVICES_UTILS_H
