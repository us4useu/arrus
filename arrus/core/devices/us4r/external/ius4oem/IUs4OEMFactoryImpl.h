#ifndef ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMFACTORYIMPL_H

#include "IUs4OEMFactory.h"

#include <ius4oem.h>
#include <numeric>

#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/us4r/external/ius4oem/Us4RLoggerWrapper.h"


namespace arrus::devices {

/**
 * A simple wrapper over GetUs4OEM method available in Us4.
 */
class IUs4OEMFactoryImpl : public IUs4OEMFactory {
public:
    IUs4OEMFactoryImpl() = default;

    IUs4OEMHandle getIUs4OEM(unsigned index) override {
        Logger::SharedHandle arrusLogger = getLoggerFactory()->getLogger();
        ::us4us::us4r::Logger::SharedHandle logger = std::make_shared<Us4RLoggerWrapper>(arrusLogger);
        return IUs4OEMHandle(GetUs4OEM(index, logger));
    }

    std::vector<IUs4OEMHandle> getModules(Ordinal nModules) override {
        std::vector<IUs4OEMHandle> us4oems;

        std::vector<Ordinal> ordinals(nModules);
        std::iota(std::begin(ordinals), std::end(ordinals), Ordinal(0));

        // Create Us4OEM handles.
        for(auto ordinal : ordinals) {
            us4oems.push_back(getIUs4OEM(ordinal));
        }
        return us4oems;
    }

    IUs4OEMFactoryImpl(IUs4OEMFactoryImpl const &) = delete;

    void operator=(IUs4OEMFactoryImpl const &) = delete;

    IUs4OEMFactoryImpl(IUs4OEMFactoryImpl const &&) = delete;

    void operator=(IUs4OEMFactoryImpl const &&) = delete;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMFACTORYIMPL_H
