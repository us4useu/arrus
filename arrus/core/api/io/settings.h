#ifndef ARRUS_CORE_API_IO_SETTINGS_H
#define ARRUS_CORE_API_IO_SETTINGS_H

#include <string>
#include "arrus/core/api/common/macros.h"
#include "arrus/core/api/session/SessionSettings.h"

namespace arrus::io {

/**
 * Reads session settings from a given file.
 *
 * The path to the file is resolved in the following way:
 * 1. use the given path (can be absolute or relative).
 * 2. if there is no regular file at the given path:
 *   1. if the provided is path is relative and ARRUS_PATH environment variable
 *      is set, use ARRUS_PATH + path.
 *   2. if there is no regular file at ARRUS_PATH + path,
 *      the IllegalArgumentException will be thrown.
 *
 * Configuration file can include a path the dictionary file, which stores
 * description of probes that have been tested by ARRUS developers.
 * The path to the dictionary file is resolved in the following way:
 *
 * 1. use the given path (can be absolute or relative).
 * 2. if there is no regular file at the given path, ry to use the parent
 *    directory of session settings file (i.e. check if the configuration file
 *    and the dictionary file (pointed by relative path) are located in the
 *    same directory.
 * 3. if 2. fails, check if the dictionary file is located at the directory
 *    pointed by ARRUS_PATH environment variable.
 * 4. if 3. fails, use default dictionary.
 */
ARRUS_CPP_EXPORT
arrus::session::SessionSettings readSessionSettings(const std::string &file);

}

#endif //ARRUS_CORE_API_IO_SETTINGS_H
