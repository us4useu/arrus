#ifndef ARRUS_ARRUS_CORE_API_SESSION_METADATA_H
#define ARRUS_ARRUS_CORE_API_SESSION_METADATA_H

#include <unordered_map>
#include <memory>
#include <utility>

namespace arrus::session {

/**
 * A container for all information related to the acquired data.
 *
 * Currently it is assumed, that the values stored in this class won't change during system run (is constant).
 */
class Metadata {
public:
    using Handle = std::unique_ptr<Metadata>;
    using SharedHandle = std::shared_ptr<Metadata>;

    explicit Metadata(std::unordered_map<std::string, std::shared_ptr<void>> metadata)
    : metadata(std::move(metadata)) {}


    /**
     * Returns metadata for the given key.
     *
     * @tparam T output type
     * @param key metadata key
     * @return metadata value for given key
     */
    template<typename T>
    std::shared_ptr<T> get(const std::string &key) {
        return std::static_pointer_cast<T>(metadata.at(key));
    }

private:
    std::unordered_map<std::string, std::shared_ptr<void>> metadata;
};

class MetadataBuilder {
public:
    MetadataBuilder() = default;

    template<typename T>
    void add(const std::string &key, std::shared_ptr<T> value) {
        metadata.emplace(key, std::move(value));
    }

    Metadata build() {
        return Metadata(metadata);
    }

    Metadata::SharedHandle buildPtr() {
        return std::make_shared<Metadata>(metadata);
    }

private:
    std::unordered_map<std::string, std::shared_ptr<void>> metadata;
};

// TODO(pjarosik) Deprecated-0.11.0
typedef Metadata UploadConstMetadata;

}

#endif //ARRUS_ARRUS_CORE_API_SESSION_METADATA_H
