#ifndef ARRUS_ARRUS_CORE_API_SESSION_UPLOADRESULT_H
#define ARRUS_ARRUS_CORE_API_SESSION_UPLOADRESULT_H

#include "arrus/core/api/framework/Buffer.h"
#include "Metadata.h"

namespace arrus::session {

/**
 * Scheme upload result.
 */
class UploadResult {
public:
    // swig
    UploadResult() {};
    virtual ~UploadResult() {};

    UploadResult(std::shared_ptr<::arrus::framework::Buffer> buffer,
                 std::shared_ptr<Metadata> constMetadata)
        : buffer(std::move(buffer)), constMetadata(std::move(constMetadata)) {}

	/**
	 * Returns a pointer to the ouptput data buffer.
	 */
    const std::shared_ptr<framework::Buffer> &getBuffer() const {
        return buffer;
    }

    /**
     * Returns a pointer to the upload constant metadata
     * (desription of the data produced by the system).
     */
    const std::shared_ptr<Metadata> &getConstMetadata() const {
        return constMetadata;
    }

private:
    ::arrus::framework::Buffer::SharedHandle buffer;
    Metadata::SharedHandle constMetadata;
};

}

#endif //ARRUS_ARRUS_CORE_API_SESSION_UPLOADRESULT_H
