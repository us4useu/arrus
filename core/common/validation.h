#ifndef ARRUS_CORE_COMMON_VALIDATION_H
#define ARRUS_CORE_COMMON_VALIDATION_H

#include <string>
#include <utility>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <sstream>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/join.hpp>

#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/common/format.h"

namespace arrus {

template<typename T>
class Validator {
public:
    explicit Validator(std::string componentName)
            : componentName(std::move(componentName)) {}

    virtual void validate(const T &obj) = 0;

    std::vector<std::string> getErrors(const std::string &parameter) {
        std::vector<std::string> result;
        auto range = errors.equal_range(parameter);
        for(auto i = range.first; i != range.second; ++i) {
            result.push_back(i->second);
        }
        return result;
    }

    bool hasErrors() {
        return !errors.empty();
    }

    void throwOnErrors() {
        if(!errors.empty()) {
            // Generate message with errors.
            std::stringstream ss;
            decltype(errors.equal_range("")) r;
            int c = 0;
            for(auto i = std::begin(errors);
                i != std::end(errors); i = r.second) {
                if(c > 0) {
                    ss << ". ";
                }
                ss << "parameter '" << i->first << "': ";
                r = errors.equal_range(i->first);
                int cc = 0;
                for(auto j = r.first; j != r.second; ++j) {
                    if(cc > 0) {
                        ss << ". ";
                    }
                    ss << j->second;
                    cc++;
                }
                c++;
            }
            std::string message = arrus::format(
                    "One or more problems have been found "
                    "with {}: {}",
                    componentName, ss.str());

            throw IllegalArgumentException(message);
        }
    }

protected:

    /**
     * Checks is given value is equal to 'expected'.
     */
    template<typename U>
    void
    expectEqual(const std::string &parameter, const U &value, const U &expected,
                const std::string &msg = "") {
        if(value != expected) {
            errors.emplace(parameter,
                           arrus::format(
                                   "Value '{}{}' should be equal '{}' "
                                   "(found: '{}')",
                                   parameter,
                                   msg,
                                   expected,
                                   value
                           ));
        }
    }

    /**
     * Checks is given value is equal to 'expected'.
     */
    template<typename U>
    void
    expectAtMost(const std::string &parameter, const U &value,
                 const U &expected, const std::string &msg = "") {
        if(value > expected) {
            errors.emplace(parameter,
                           arrus::format(
                                   "Value '{}{}' should be at most '{}' "
                                   "(found: '{}')",
                                   parameter, msg, expected, value
                           ));
        }
    }

    /**
     * Checks is given value is divisible by divider.
     */
    template<typename U>
    void
    expectDivisible(
            const std::string &parameter, const U &value, const U &divider,
            const std::string &msg = "") {
        if(value % divider != 0) {
            errors.emplace(parameter,
                           arrus::format(
                                   "Value '{}{}' should be divisible by '{}' "
                                   "(actually is: '{}')",
                                   parameter,
                                   msg,
                                   divider,
                                   value
                           ));
        }
    }

    /**
     * Checks if all given values are in range [min, max].
     */
    template<typename U>
    void
    expectAllInRange(const std::string &parameter, const std::vector<U> &values,
                     const U &min, const U &max, const std::string &msg = "") {

        std::set<U> invalidValues;

        for(auto value : values) {
            if(!(value >= min && value <= max)) {
                invalidValues.insert(value);
            }
        }

        if(!invalidValues.empty()) {
            errors.emplace(
                    parameter,
                    arrus::format("Value(s) '{}{}' should be in range [{}, {}] "
                                  "(found: '{}')",
                                  parameter, msg,
                                  min, max, toString(invalidValues)
                    )
            );
        }
    }

    template<typename U>
    void
    expectAllPositive(const std::string &parameter,
                      const std::vector<U> &values) {
        std::set<U> invalidValues;
        for(auto value : values) {
            if(value <= 0) {
                invalidValues.insert(value);
            }
        }
        if(!invalidValues.empty()) {
            errors.emplace(
                    parameter,
                    arrus::format("Value(s) '{}' should be positive "
                                  "(found: '{}')",
                                  parameter, toString(invalidValues)
                    )
            );
        }
    }

    /**
     * Checks if given value is in range [min, max].
     */
    template<typename U>
    void
    expectInRange(const std::string &parameter, const U &value, const U &min,
                  const U &max, const std::string &msg = "") {
        if(!(value >= min && value <= max)) {
            errors.emplace(parameter,
                           arrus::format(
                                   "Value '{}{}' should be in range [{}, {}] "
                                   "(found: '{}')",
                                   parameter,
                                   msg,
                                   min, max,
                                   value
                           ));
        }
    }

    template<typename U, typename Container>
    void
    expectOneOf(const std::string &parameter, U value, Container dictionary,
                const std::string &msg = "") {
        if(dictionary.find(value) == dictionary.end()) {
            // Concatenate and sort dictionary values.
            std::vector<std::string> stringRepresentation;
            std::transform(std::begin(dictionary), std::end(dictionary),
                           std::back_inserter(stringRepresentation),
                           [](auto &val) {
                               return boost::lexical_cast<std::string>((U) val);
                           });
            errors.emplace(parameter, arrus::format(
                    "Value '{}{}' should be one of: '{}' (found: '{}')",
                    parameter,
                    msg,
                    boost::algorithm::join(stringRepresentation, ", "),
                    value
            ));
        }
    }

    void expectTrue(const std::string &parameter,
                    bool condition, const std::string &msg) {
        if(!condition) {
            errors.emplace(parameter, arrus::format(
                    "Value '{}': {}",
                    parameter, msg));
        }
    }

    void expectFalse(const std::string &parameter,
                     bool condition, const std::string &msg) {
        if(condition) {
            errors.emplace(parameter, arrus::format(
                    "Value '{}': {}",
                    parameter, msg));
        }
    }


private:
    // parameter name -> messages
    std::multimap<std::string, std::string> errors;
    std::string componentName;
};
}

#endif //ARRUS_CORE_COMMON_VALIDATION_H
