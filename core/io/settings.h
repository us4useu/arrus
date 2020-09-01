#ifndef ARRUS_CORE_IO_SETTINGS_H
#define ARRUS_CORE_IO_SETTINGS_H

#include "arrus/core/session/SessionSettings.h"

namespace arrus::io {

SessionSettings readSessionSettings(const std::string& file) {
//    return nullptr;
    throw IllegalStateException("NYI");
}

}

#endif //ARRUS_CORE_IO_SETTINGS_H
