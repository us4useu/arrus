#ifndef ARRUS_CORE_API_COMMON_PARAMETERS_H
#define ARRUS_CORE_API_COMMON_PARAMETERS_H

#include <unordered_map>
#include <utility>
#include "arrus/core/api/common/exceptions.h"

namespace arrus {

/**
 * A container for key-value parameters.
 * This class is immutable.
 */
class Parameters {
public:
    using Handle = std::unique_ptr<Parameters>;
    using SharedHandle = std::shared_ptr<Parameters>;

    explicit Parameters(std::unordered_map<std::string, int> values): values(std::move(values)) {}

    /**
     * Returns value for parameter with the given key.
     *
     * @param key parameter key
     * @return parameter value
     */
    int get(const std::string &key) {
        try{
            return values.at(key);
        } catch (const std::out_of_range &) {
            throw arrus::IllegalArgumentException("Parameter unavailable: " + key);
        }
    }

    const std::unordered_map<std::string, int> &items() const { return values; }

private:
    std::unordered_map<std::string, int> values;
};

class ParametersBuilder {
public:
    ParametersBuilder() = default;

    void add(const std::string &key, int value) {
        values.emplace(key, value);
    }

    Parameters build() {
        return Parameters(values);
    }

private:
    std::unordered_map<std::string, int> values;
};

}




#endif//ARRUS_CORE_API_COMMON_PARAMETERS_H
