#ifndef CPP_EXAMPLE_METADATA_H
#define CPP_EXAMPLE_METADATA_H

#include <memory>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace arrus_example_imaging {

class MetadataBuilder;

class Metadata {
public:
    using Handle = std::unique_ptr<Metadata>;
    using SharedHandle = std::shared_ptr<Metadata>;

    Metadata() = default;

    explicit Metadata(std::unordered_map<std::string, std::shared_ptr<void>> ptrs,
                      std::unordered_map<std::string, float> values)
        : ptrs(std::move(ptrs)), values(std::move(values)) {}

    /**
     * Returns metadata for the given key.
     *
     * @tparam T output type
     * @param key metadata key
     * @return metadata value for given key
     */
    template<typename T> std::shared_ptr<T> getObject(const std::string &key) {
        try {
            return std::static_pointer_cast<T>(ptrs.at(key));
        } catch (std::out_of_range &e) { throw std::out_of_range{"There is no object in metadata with key: " + key}; }
    }

    float getValue(const std::string &key) {
        try {
            return values.at(key);
        } catch (std::out_of_range &e) { throw std::out_of_range{"There is no value in metadata with key: " + key}; }
    }

private:
    friend class MetadataBuilder;
    std::unordered_map<std::string, std::shared_ptr<void>> ptrs;
    std::unordered_map<std::string, float> values;
};

class MetadataBuilder {
public:
    MetadataBuilder() = default;

    explicit MetadataBuilder(const Metadata &metadata) : ptrs(metadata.ptrs), values(metadata.values) {}
    explicit MetadataBuilder(const std::shared_ptr<Metadata> &metadata)
        : ptrs(metadata->ptrs), values(metadata->values) {}

    template<typename T> std::shared_ptr<T> getObject(const std::string &key) {
        try {
            return std::static_pointer_cast<T>(ptrs.at(key));
        } catch (std::out_of_range &e) { throw std::out_of_range{"There is no object in metadata with key: " + key}; }
    }

    float getValue(const std::string &key) {
        try {
            return values.at(key);
        } catch (std::out_of_range &e) { throw std::out_of_range{"There is no value in metadata with key: " + key}; }
    }

    template<typename T> void addObject(const std::string &key, const std::shared_ptr<T> &ptr) {
        ptrs[key] = std::static_pointer_cast<void>(ptr);
    }

    void setValue(const std::string &key, float value) { values[key] = value; }

    Metadata build() { return Metadata(ptrs, values); }

    std::shared_ptr<Metadata> buildSharedPtr() { return std::make_shared<Metadata>(ptrs, values); }

private:
    std::unordered_map<std::string, std::shared_ptr<void>> ptrs;
    std::unordered_map<std::string, float> values;
};

}// namespace arrus_example_imaging

#endif
