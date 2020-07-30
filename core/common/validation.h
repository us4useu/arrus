#ifndef ARRUS_CORE_COMMON_VALIDATION_H
#define ARRUS_CORE_COMMON_VALIDATION_H

#include <string>
#include <vector>

#include "core/common/exceptions.h"

namespace arrus {

template<typename T>
class Validator {
public:

    explicit Validator(const T &obj) {
        validate(obj);
    }

    virtual void validate(const T &obj) = 0;

    void throwOnErrors() {
        if(!errors.empty()) {
            throw IllegalArgumentException("");
        }
    }

protected:
    void expectTrue(bool condition, const std::string &errorMsg) {
        if(!condition) {
            errors.push_back(errorMsg);
        }
    }


private:
    std::vector<std::string> errors;
};
}

#endif //ARRUS_CORE_COMMON_VALIDATION_H
