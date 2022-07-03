#ifndef ARRUS_ARRUS_CORE_API_SESSION_UPLOADRESULT_H
#define ARRUS_ARRUS_CORE_API_SESSION_UPLOADRESULT_H

#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/api/framework/Metadata.h"

namespace arrus::session {

/**
 * Scheme upload result.
 */
class UploadResult {
public:
    // swig
    UploadResult() {};
    virtual ~UploadResult() {};

    UploadResult(std::shared_ptr<::arrus::framework::Buffer> buffer, Metadata metadata)
        : buffer(std::move(buffer)), metadata(std::move(metadata)) {}

	/**
	 * Returns a pointer to the ouptput data buffer.
	 */
    const std::shared_ptr<framework::Buffer> &getBuffer() const {
        return buffer;
    }

    /**
     * Returns a pointer to the upload  metadata (description of the data produced by the system).
     *
     * @deprecated use getMetadata()
     */
    const Metadata &getConstMetadata() const {
        return getMetadata();
    }

    const Metadata &getMetadata() const {
        return metadata;
    }

private:
    ::arrus::framework::Buffer::SharedHandle buffer;
    Metadata metadata;
};

}

#endif //ARRUS_ARRUS_CORE_API_SESSION_UPLOADRESULT_H
