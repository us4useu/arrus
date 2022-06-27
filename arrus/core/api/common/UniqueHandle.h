#ifndef ARRUS_CORE_API_COMMON_UNIQUEHANDLE_H
#define ARRUS_CORE_API_COMMON_UNIQUEHANDLE_H

#include <memory>

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
class ARRUS_CPP_EXPORT UniqueHandle {
public:
    template<typename... Args>
    static UniqueHandle<T> create(Args&&... args) {
        return UniqueHandle<T>{new T(std::forward<Args>(args)...)};
    }

    UniqueHandle(nullptr_t v = nullptr): ptr{v} {}

    UniqueHandle(const UniqueHandle &other) {
        if(other) {
            ptr = new T{*other};
        }
    }

    UniqueHandle(UniqueHandle &&other)  noexcept {
        this->swap(other);
    }

    UniqueHandle& operator=(const UniqueHandle &other) {
        UniqueHandle<T> copy{other};
        this->swap(copy);
        return *this;
    }

    UniqueHandle& operator=(UniqueHandle&& other)  noexcept {
        this->swap(other);
        return *this;
    }

    void swap(UniqueHandle& r) noexcept {std::swap(ptr, r.ptr);}

    const T* get() const { return ptr; }

    T* get() { return ptr; }

    T& operator*() { return *get(); }

    const T& operator*() const { return *get(); }

    T* operator->() { return get(); }

    const T* operator->() const { return get(); }

    explicit operator bool() const { return (bool)ptr; }

private:
    UniqueHandle(T* ptr): ptr(ptr) {}
    T* ptr{nullptr};
};

}

#endif//ARRUS_CORE_API_COMMON_UNIQUEHANDLE_H
