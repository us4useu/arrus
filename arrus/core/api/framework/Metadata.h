#ifndef ARRUS_CORE_API_FRAMEWORK_METADATA_H
#define ARRUS_CORE_API_FRAMEWORK_METADATA_H

#include <unordered_map>
#include <memory>
#include <utility>

#include "arrus/core/api/common/UniqueHandle.h"
#include "arrus/core/api/common/String.h"

namespace arrus::framework {

/**
 * A container for all information related to the processed data.
 *
 * Note: this class is immutable.
 * Currently it is assumed, that the values stored in this class won't change during system run (is constant).
 */
class Metadata {
    class Impl;
    ::arrus::UniqueHandle<Impl> impl;
public:
    Metadata();
    Metadata(const Metadata &other);
    Metadata(Metadata &&other) noexcept;
    Metadata &operator=(const Metadata &other);
    Metadata &operator=(Metadata &&other) noexcept;
    ~Metadata();

    // TODO(pjarosik) do not use std::shared_ptr in the interface
    /**
     * Returns metadata for the given key.
     *
     * @tparam T output type
     * @param key metadata key
     * @return metadata value for given key
     */
    template<typename T>
    std::shared_ptr<T> get(const String &key) {
        return std::static_pointer_cast<T>(getObject(key));
    }

    /**
     * Returns a new instance of Metadata object with the given key set to the
     * @tparam T
     * @param key
     * @param value
     * @return
     */
    template<typename T>
    Metadata set(const String &key, std::shared_ptr<T> value) {
        return setObject(key, std::move(std::static_pointer_cast<void>(value)));
    }

    std::shared_ptr<void> getObject(const String &key);

    Metadata setObject(const String &key, std::shared_ptr<void> value);
};

}

#endif //ARRUS_CORE_API_FRAMEWORK_METADATA_H
