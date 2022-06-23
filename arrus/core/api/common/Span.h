#ifndef ARRUS_CORE_API_COMMON_SPAN_H
#define ARRUS_CORE_API_COMMON_SPAN_H

#include <cstddef>

namespace arrus {

/**
 * Note: this class does not own the underlying data (view).
 * @tparam T
 */
template<typename T>
class Span {
public:
    Span(T *ptr, size_t nElements) : ptr(ptr), nElements(nElements) {}

    Span(const std::vector<T> vec): ptr(vec.data()), nElements(vec.size()) {}

    const T* data() const noexcept {
        return ptr;
    }

    size_t size() const noexcept {
        return nElements;
    }

    const T& operator[](size_t i) const {
        return ptr[i];
    }

private:
    const T* ptr;
    /** Number of T elements */
    size_t nElements;
};

}

#endif//ARRUS_CORE_API_COMMON_SPAN_H
