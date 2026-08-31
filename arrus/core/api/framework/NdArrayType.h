#ifndef ARRUS_CORE_API_FRAMEWORK_NDARRAYTYPE_H
#define ARRUS_CORE_API_FRAMEWORK_NDARRAYTYPE_H

#include <functional>
#include <ostream>
#include <utility>

#include "arrus/core/api/framework/DType.h"
#include "arrus/core/api/framework/DeviceRef.h"
#include "arrus/core/api/framework/Shape.h"

namespace arrus::framework {

/**
 * Type descriptor for an NdArray or NdStorage: (shape, dtype, device).
 *
 * Shared between:
 *   - NdArray (lazy graph-node)   -- shape may contain unknown dims
 *                                    until shape/type inference runs.
 *   - NdStorage (concrete memory) -- shape must be fully known.
 *
 * The device may be "ANY" in an authored graph; the compiler's
 * placement pass resolves it to a concrete DeviceRef.
 *
 * Immutable value class. Compare, hash, print.
 */
class NdArrayType {
public:
    NdArrayType() = default;

    NdArrayType(Shape shape, DType dtype, DeviceRef device = DeviceRef{})
        : shape_(std::move(shape)),
          dtype_(dtype),
          device_(std::move(device)) {}

    const Shape &shape() const { return shape_; }
    DType dtype() const { return dtype_; }
    const DeviceRef &device() const { return device_; }

    /**
     * True iff every shape dim is known. Independent of the placement
     * status of `device_` (a fully-known shape may still be placed on
     * "ANY").
     */
    bool isFullyKnown() const { return shape_.isFullyKnown(); }

    bool operator==(const NdArrayType &other) const {
        return dtype_ == other.dtype_
            && shape_ == other.shape_
            && device_ == other.device_;
    }
    bool operator!=(const NdArrayType &other) const { return !(*this == other); }

    friend std::ostream &operator<<(std::ostream &os, const NdArrayType &t) {
        return os << dtypeName(t.dtype_) << t.shape_ << "@" << t.device_;
    }

private:
    Shape shape_;
    DType dtype_{DType::UNKNOWN};
    DeviceRef device_{}; // default "ANY"
};

} // namespace arrus::framework

namespace std {
template<>
struct hash<::arrus::framework::NdArrayType> {
    size_t operator()(const ::arrus::framework::NdArrayType &t) const noexcept {
        size_t h = std::hash<uint32_t>{}(static_cast<uint32_t>(t.dtype()));
        h ^= std::hash<::arrus::framework::DeviceRef>{}(t.device())
             + 0x9e3779b9 + (h << 6) + (h >> 2);
        for (const auto &d : t.shape().dims()) {
            size_t dh = d.has_value() ? std::hash<size_t>{}(*d) : 0xffffffff;
            h ^= dh + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;
    }
};
} // namespace std

#endif // ARRUS_CORE_API_FRAMEWORK_NDARRAYTYPE_H
