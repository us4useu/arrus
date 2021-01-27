#ifndef ARRUS_ARRUS_CORE_API_SESSION_UPLOADCONSTMETADATA_H
#define ARRUS_ARRUS_CORE_API_SESSION_UPLOADCONSTMETADATA_H

#include <unordered_map>
#include <memory>
#include <utility>

namespace arrus::session {

/**
 * A container for all information related to the acquired data.
 *
 * Currently it is assumed, that the values stored in this class won't change during system run (is constant).
 */
class UploadConstMetadata {
public:
    using Handle = std::unique_ptr<UploadConstMetadata>;
    using SharedHandle = std::shared_ptr<UploadConstMetadata>;

    explicit UploadConstMetadata(std::unordered_map<std::string, std::shared_ptr<void>> metadata)
    : metadata(std::move(metadata)) {}


    template<typename T>
    std::shared_ptr<T> get(const std::string &key) {
        return std::static_pointer_cast<T>(metadata.at(key));
    }

private:
    std::unordered_map<std::string, std::shared_ptr<void>> metadata;
};

}

#endif //ARRUS_ARRUS_CORE_API_SESSION_UPLOADCONSTMETADATA_H
