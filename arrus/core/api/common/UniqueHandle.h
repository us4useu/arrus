#ifndef ARRUS_CORE_API_COMMON_UNIQUEHANDLE_H
#define ARRUS_CORE_API_COMMON_UNIQUEHANDLE_H

#include <memory>
#include <experimental/propagate_const>

namespace arrus {

/**
 * A wrapper for std::unique pointer, that:
 * - performs deep-copy on copy construction,
 * - ordinary pointer movement on move construction.
 * - swap on assignment.
 *
 * NOTE: this class will not work properly when assigning derived class pointers to a base class pointer.
 * This is because we use T constructor to do all the deep copy and assignments.
 */
template<typename T>
class UniqueHandle {
public:
    template<typename... Args>
    static UniqueHandle<T> create(Args&&... args) {
        return UniqueHandle<T>{std::move(std::make_unique<T>(std::forward<Args>(args)...))};
    }

    static void swap(UniqueHandle& l, UniqueHandle& r) noexcept {std::swap(l.ptr, r.ptr);}

    UniqueHandle(nullptr_t v = nullptr): ptr{v} {}

    UniqueHandle(const UniqueHandle &other) {
        if(other) {
            ptr = std::unique_ptr<T>{new T{*other}};
        }
    }

    UniqueHandle(UniqueHandle &&other)  noexcept {
        std::swap(*this, other);
    }

    UniqueHandle& operator=(const UniqueHandle &other) {
        UniqueHandle<T> copy{other};
        swap(*this, copy);
        return *this;
    }

    UniqueHandle& operator=(UniqueHandle&& other)  noexcept {
        swap(*this, other);
        return *this;
    }

    T& operator*() { return *ptr; }

    const T& operator*() const { return *ptr; }

    T* operator->() { return ptr.operator->(); }

    const T* operator->() const { return ptr.operator->(); }

    const T* get() const { return ptr.get(); }

    explicit operator bool() const { return (bool)ptr; }

private:
    explicit UniqueHandle(std::unique_ptr<T> v): ptr(std::move(v)) {}

    std::experimental::propagate_const<std::unique_ptr<T>> ptr{nullptr};
};

}

#endif//ARRUS_CORE_API_COMMON_UNIQUEHANDLE_H
