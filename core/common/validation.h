#ifndef ARRUS_CORE_COMMON_VALIDATION_H
#define ARRUS_CORE_COMMON_VALIDATION_H

#include <string>
#include <vector>
#include <unordered_set>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/join.hpp>

#include "arrus/core/common/exceptions.h"
#include "arrus/core/common/format.h"

namespace arrus {

template<typename T>
class Validator {
public:
    virtual void validate(const T &obj) = 0;

    void throwOnErrors() {
        if (!errors.empty()) {
            throw IllegalArgumentException();
        }
    }

protected:

    /**
     * Checks if given value is in range [min, max].
     */
    template<typename U>
    void
    expectInRange(const U &value, const U &min, const U &max,
                  const std::string &valueName) {
        if (!(value >= min && value <= max)) {
            errors.push_back(
                    arrus::format(
                            "Value '{}' should be in range [{}, {}] "
                            "(found: '{}')",
                            valueName,
                            min, max,
                            value
                    ));
        }
    }

    template<typename U, typename Container>
    void
    expectOneOf(U value, Container dictionary, const std::string &valueName) {
        if (dictionary.find(value) == dictionary.end()) {
            // Concatenate and sort dictionary values.
            std::vector<std::string> stringRepresentation;
            std::transform(std::begin(dictionary), std::end(dictionary),
                           std::back_inserter(stringRepresentation),
                           [](auto &val) {
                               return boost::lexical_cast<std::string>(val);
                           });
            errors.push_back(arrus::format(
                    "Value '{}' should be one of: '{}' (found: '{}')",
                    valueName,
                    boost::algorithm::join(stringRepresentation, ", "),
                    value
            ));
        }
    }

    void expectTrue(bool condition, const std::string &errorMsg) {
        if (!condition) {
            errors.push_back(errorMsg);
        }
    }


private:
    std::vector<std::string> errors;
};
}

#endif //ARRUS_CORE_COMMON_VALIDATION_H
