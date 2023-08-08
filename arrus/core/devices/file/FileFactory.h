#ifndef ARRUS_CORE_DEVICES_FILE_FILEFACTORY_H
#define ARRUS_CORE_DEVICES_FILE_FILEFACTORY_H

namespace arrus::devices {

class FileFactory {
public:
    using Handle = std::unique_ptr<FileFactory>;

    virtual File::Handle getFile(Ordinal ordinal, const FileSettings &settings) = 0;

    virtual ~FileFactory() = default;
};

}

#endif//ARRUS_CORE_DEVICES_FILE_FILEFACTORY_H
