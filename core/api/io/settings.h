#ifndef ARRUS_CORE_API_IO_SETTINGS_H
#define ARRUS_CORE_API_IO_SETTINGS_H

#include <string>
#include "arrus/core/api/session/SessionSettings.h"

namespace arrus::io {

SessionSettings readSessionSettings(const std::string &file);

}

#endif //ARRUS_CORE_API_IO_SETTINGS_H
