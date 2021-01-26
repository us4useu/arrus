#ifndef ARRUS_ARRUS_CORE_API_SESSION_UPLOADRESULT_H
#define ARRUS_ARRUS_CORE_API_SESSION_UPLOADRESULT_H

#include "arrus/core/api/framework/FifoBuffer.h"
#include "UploadConstMetadata.h"

namespace arrus::session {

class UploadResult {
public:
    UploadResult(::arrus::framework::FifoBuffer::SharedHandle buffer, UploadConstMetadata::SharedHandle constMetadata)
        : buffer(std::move(buffer)), constMetadata(std::move(constMetadata)) {}

    const framework::FifoBuffer::SharedHandle &getBuffer() const {
        return buffer;
    }

    const UploadConstMetadata::SharedHandle &getConstMetadata() const {
        return constMetadata;
    }

private:
    ::arrus::framework::FifoBuffer::SharedHandle buffer;
    UploadConstMetadata::SharedHandle constMetadata;
};

}

#endif //ARRUS_ARRUS_CORE_API_SESSION_UPLOADRESULT_H
