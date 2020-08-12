#ifndef ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMFACTORYIMPL_H

#include "IUs4OEMFactory.h"

#include <ius4oem.h>

#include "arrus/core/common/logging.h"
#include "arrus/core/devices/us4r/external/ius4oem/Us4RLoggerWrapper.h"


namespace arrus {

/**
 * A simple wrapper over GetUs4OEM method available in Us4.
 */
class IUs4OEMFactoryImpl : public IUs4OEMFactory {
public:
    static IUs4OEMFactory &getInstance() {
        static IUs4OEMFactoryImpl instance;
        return instance;
    }

    IUs4OEMHandle getIUs4OEM(unsigned index) override {
        us4r::Logger::SharedHandle logger =
                std::make_shared<Us4RLoggerWrapper>(getDefaultLogger());
        return IUs4OEMHandle(GetUs4OEM(index, logger));
    }

    IUs4OEMFactoryImpl(IUs4OEMFactoryImpl const &) = delete;

    void operator=(IUs4OEMFactoryImpl const &) = delete;

    IUs4OEMFactoryImpl(IUs4OEMFactoryImpl const &&) = delete;

    void operator=(IUs4OEMFactoryImpl const &&) = delete;

private:
    IUs4OEMFactoryImpl() = default;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMFACTORYIMPL_H
