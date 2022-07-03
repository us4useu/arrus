#include "arrus/core/api/framework/Metadata.h"

#include <memory>
#include <unordered_map>
#include <utility>

namespace arrus::framework {

class Metadata::Impl {
public:
    Impl() = default;

    std::shared_ptr<void> getObject(const String &key){
        return metadata.at(key.copyToString());
    }

    void set(const String &key, std::shared_ptr<void> value) {
        metadata.insert_or_assign(key.copyToString(), value);
    }

private:
    std::unordered_map<std::string, std::shared_ptr<void>> metadata;
};

Metadata::Metadata() { this->impl = UniqueHandle<Metadata::Impl>::create(); }
Metadata::~Metadata() = default;
Metadata::Metadata(Metadata &&metadata) noexcept = default;
Metadata::Metadata(const Metadata &metadata) = default;
Metadata &Metadata::operator=(const Metadata &other) = default;
Metadata &Metadata::operator=(Metadata &&other) noexcept = default;

std::shared_ptr<void> Metadata::getObject(const String &key) { return this->impl->getObject(key); }

Metadata Metadata::setObject(const String &key, std::shared_ptr<void> value) {
    Metadata newMetadata{*this};
    newMetadata.impl->set(key, std::move(value));
    return newMetadata;
}


}// namespace arrus::session