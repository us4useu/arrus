#ifndef ARRUS_CORE_DEVICES_FILE_FILEFACTORY_H
#define ARRUS_CORE_DEVICES_FILE_FILEFACTORY_H

#include "arrus/core/api/devices/File.h"
#include "arrus/core/api/devices/FileSettings.h"

namespace arrus::devices {

class FileFactory {
public:
    using Handle = std::unique_ptr<FileFactory>;

    virtual File::Handle getFile(Ordinal ordinal, const FileSettings &settings) = 0;

    virtual ~FileFactory() = default;
};

}

#endif//ARRUS_CORE_DEVICES_FILE_FILEFACTORY_H
