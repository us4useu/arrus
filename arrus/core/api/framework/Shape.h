#ifndef ARRUS_CORE_API_FRAMEWORK_SHAPE_H
#define ARRUS_CORE_API_FRAMEWORK_SHAPE_H

#include <cstddef>
#include <initializer_list>
#include <optional>
#include <ostream>
#include <utility>
#include <vector>

#include "arrus/core/api/common/exceptions.h"

namespace arrus::framework {

/**
 * Shape of an NdArray or NdStorage value.
 *
 * Each dimension is a std::optional<size_t>: the value is present when
 * the dimension is known, absent when it is not (typically before the
 * shape/type-inference pass resolves it). NdStorage instances must have
 * a fully-known shape (isFullyKnown() == true); NdArray instances may
 * carry unknown dims until inference completes.
 *
 * Immutable value class.
 */
class Shape {
public:
    using Dim = std::optional<size_t>;

    Shape() = default;
    Shape(std::initializer_list<Dim> dims) : dims_(dims) {}
    explicit Shape(std::vector<Dim> dims) : dims_(std::move(dims)) {}

    /** Number of dimensions. Zero means the shape describes a scalar. */
    size_t rank() const { return dims_.size(); }

    /** True iff rank() == 0. */
    bool isScalar() const { return dims_.empty(); }

    /** True iff every dimension is known. */
    bool isFullyKnown() const {
        for (const auto &d : dims_) {
            if (!d.has_value()) { return false; }
        }
        return true;
    }

    /** i-th dimension. UB if i >= rank(). */
    Dim operator[](size_t i) const { return dims_[i]; }

    /** All dimensions as a vector. */
    const std::vector<Dim> &dims() const { return dims_; }

    /**
     * Product of dimensions. Requires isFullyKnown(); throws
     * IllegalArgumentException otherwise. Returns 1 for a scalar shape.
     */
    size_t numElements() const {
        size_t n = 1;
        for (const auto &d : dims_) {
            if (!d.has_value()) {
                throw IllegalArgumentException(
                    "numElements() on a shape with unknown dims");
            }
            n *= *d;
        }
        return n;
    }

    bool operator==(const Shape &other) const { return dims_ == other.dims_; }
    bool operator!=(const Shape &other) const { return !(*this == other); }

    friend std::ostream &operator<<(std::ostream &os, const Shape &s) {
        os << "(";
        for (size_t i = 0; i < s.dims_.size(); ++i) {
            if (i > 0) { os << ", "; }
            if (s.dims_[i].has_value()) {
                os << *s.dims_[i];
            } else {
                os << "?";
            }
        }
        os << ")";
        return os;
    }

private:
    std::vector<Dim> dims_;
};

} // namespace arrus::framework

#endif // ARRUS_CORE_API_FRAMEWORK_SHAPE_H
