#ifndef ARRUS_CORE_API_FRAMEWORK_DEVICEREF_H
#define ARRUS_CORE_API_FRAMEWORK_DEVICEREF_H

#include <functional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>

namespace arrus::framework {

/**
 * Abstract placement target for an NdArray or NdStorage value.
 *
 * A DeviceRef is a small string in one of the following forms:
 *
 *     "ANY"            placement unresolved; the compiler chooses at
 *                      compile time. Default in user-authored graphs.
 *     "CPU"            host CPU
 *     "GPU:0"          CUDA GPU with ordinal 0
 *     "Us4R:0"         Us4R device with ordinal 0
 *     "Us4R:0/OEM:2"   nested placement into a composite device
 *
 * DeviceRef is a value class: small, copy-cheap, hashable, printable.
 * It does not interpret the string; interpretation (resolution to a
 * concrete backend) happens in the placement pass of the compiler.
 */
class DeviceRef {
public:
    /** The literal "ANY" placement token. */
    static constexpr std::string_view ANY_VALUE = "ANY";

    /** Constructs a DeviceRef with the default "ANY" placement. */
    DeviceRef() : value_(ANY_VALUE) {}

    explicit DeviceRef(std::string value) : value_(std::move(value)) {}
    explicit DeviceRef(std::string_view value) : value_(value) {}
    explicit DeviceRef(const char *value) : value_(value) {}

    /** The underlying string. */
    const std::string &value() const { return value_; }

    /** True iff this is the "ANY" placement. */
    bool isAny() const { return value_ == ANY_VALUE; }

    bool operator==(const DeviceRef &other) const { return value_ == other.value_; }
    bool operator!=(const DeviceRef &other) const { return !(*this == other); }

    friend std::ostream &operator<<(std::ostream &os, const DeviceRef &d) {
        return os << d.value_;
    }

private:
    std::string value_;
};

} // namespace arrus::framework

namespace std {
template<>
struct hash<::arrus::framework::DeviceRef> {
    size_t operator()(const ::arrus::framework::DeviceRef &d) const noexcept {
        return std::hash<std::string>{}(d.value());
    }
};
} // namespace std

#endif // ARRUS_CORE_API_FRAMEWORK_DEVICEREF_H
