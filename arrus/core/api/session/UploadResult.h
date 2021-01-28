#ifndef ARRUS_ARRUS_CORE_API_SESSION_UPLOADRESULT_H
#define ARRUS_ARRUS_CORE_API_SESSION_UPLOADRESULT_H

#include "arrus/core/api/framework/DataBuffer.h"
#include "UploadConstMetadata.h"

namespace arrus::session {

class UploadResult {
public:
    // swig
    UploadResult() {};
    virtual ~UploadResult() {};

    UploadResult(std::shared_ptr<::arrus::framework::DataBuffer> buffer,
                 std::shared_ptr<UploadConstMetadata> constMetadata)
        : buffer(std::move(buffer)), constMetadata(std::move(constMetadata)) {}

    const std::shared_ptr<framework::DataBuffer> &getBuffer() const {
        return buffer;
    }

    const std::shared_ptr<UploadConstMetadata> &getConstMetadata() const {
        return constMetadata;
    }

private:
    ::arrus::framework::DataBuffer::SharedHandle buffer;
    UploadConstMetadata::SharedHandle constMetadata;
};

}

#endif //ARRUS_ARRUS_CORE_API_SESSION_UPLOADRESULT_H
