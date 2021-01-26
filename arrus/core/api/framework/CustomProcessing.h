#ifndef ARRUS_CORE_API_FRAMEWORK_H
#define ARRUS_CORE_API_FRAMEWORK_H

namespace arrus::framework {

class CustomProcessing {

    virtual ~CustomProcessing = default;

    virtual prepare(const PreparationContext& context) = 0;

    virtual process(const ProcessingContetx& context) = 0;
};

}

#endif //ARRUS_CORE_API_FRAMEWORK_H
