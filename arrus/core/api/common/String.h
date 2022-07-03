#ifndef ARRUS_CORE_API_COMMON_STRING_H
#define ARRUS_CORE_API_COMMON_STRING_H

#include <cstring>
#include <string>
#include <utility>

namespace arrus {

class String {
public:
    String() = default;
    /**
     * Copies input string to internal representation.
     *
     * @param str input string, should be null-terminated.
     */
    String(const char *str) { set(str); }

    String(const std::string &str) : String(str.c_str()) {}

    String(const String &other) { set(other.str); }

    String(String &&other) noexcept : String() { swap(*this, other); }

    virtual ~String() {
        delete[] str;
        this->str = nullptr;
        this->len = 0;
    }

    String &operator=(String other) {
        swap(*this, other);
        return *this;
    }

    String &operator=(String &&other) noexcept {
        swap(*this, other);
        return *this;
    }

    friend void swap(String &a, String &b) {
        using std::swap;
        swap(a.str, b.str);
        swap(a.len, b.len);
    }

    std::string copyToString() const { return std::string{str}; }

    std::string_view toStringView() const { return std::string_view{str}; }

private:
    void set(const char *s) {
        this->len = strlen(s);
        if (this->len == 0) {
            return;
        }
        this->str = new char[len];
        std::strcpy(this->str, s);
    }

    char *str{nullptr};
    size_t len{0};
};

/**
 * ARRUS implementation of the string view object. The objects of this class are immutable.
 * The objects of this class does not own the provided parameter.
 */
class StringView {
public:
    StringView(const char *str) : str(str) {}
    StringView(std::string_view stringView) : str(stringView.data()) {}

    operator std::string_view() const { return std::string_view{str}; }

private:
    const char *str;
};

}// namespace arrus

#endif//ARRUS_CORE_API_COMMON_STRING_H
