#ifndef ARRUS_CORE_DEVICES_FILE_FILEFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_FILE_FILEFACTORYIMPL_H

#include "arrus/core/devices/file/FileFactory.h"
#include "arrus/core/api/devices/File.h"
#include "arrus/core/devices/file/FileImpl.h"

namespace arrus::devices {

class FileFactoryImpl: public FileFactory {

public:
    File::Handle getFile(Ordinal ordinal, const FileSettings &settings) override {
        DeviceId id(DeviceType::File, ordinal);
        return std::make_unique<FileImpl>(id, settings);
    }
};


}

#endif//ARRUS_CORE_DEVICES_FILE_FILEFACTORYIMPL_H
