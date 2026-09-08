#ifndef ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMFACTORYIMPL_H

#include "IUs4OEMFactory.h"

#include <array>
#include <ius4oem.h>
#include <numeric>
#include <utility>

#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/devices/us4r/Us4RSettings.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/us4r/external/ius4oem/Us4RLoggerWrapper.h"


namespace arrus::devices {

/**
 * A simple wrapper over GetUs4OEM method available in Us4.
 */
class IUs4OEMFactoryImpl : public IUs4OEMFactory {
public:
    IUs4OEMFactoryImpl() = default;

    IUs4OEMHandle getIUs4OEM(unsigned index,
                             const Us4OEMInterruptCallbacksMap &callbacks = {}) override {
        Logger::SharedHandle arrusLogger = getLoggerFactory()->getLogger();
        ::us4us::us4r::Logger::SharedHandle logger = std::make_shared<Us4RLoggerWrapper>(arrusLogger);
        IUs4OEM::CallbacksMap msiCallbacks = buildMSICallbacks(index, callbacks);
        return IUs4OEMHandle(GetUs4OEM(index, logger, msiCallbacks));
    }

    std::vector<IUs4OEMHandle> getModules(Ordinal nModules,
                                          const Us4OEMInterruptCallbacksMap &callbacks = {}) override {
        std::vector<IUs4OEMHandle> us4oems;

        std::vector<Ordinal> ordinals(nModules);
        std::iota(std::begin(ordinals), std::end(ordinals), Ordinal(0));

        // Create Us4OEM handles.
        for(auto ordinal : ordinals) {
            us4oems.push_back(getIUs4OEM(ordinal, callbacks));
        }
        return us4oems;
    }

    IUs4OEMFactoryImpl(IUs4OEMFactoryImpl const &) = delete;

    void operator=(IUs4OEMFactoryImpl const &) = delete;

    IUs4OEMFactoryImpl(IUs4OEMFactoryImpl const &&) = delete;

    void operator=(IUs4OEMFactoryImpl const &&) = delete;

private:
    /**
     * Builds an IUs4OEM::CallbacksMap by registering, for each entry in the
     * user-provided callbacks map, a wrapper that invokes that specific
     * callback with the OEM ordinal.
     */
    static IUs4OEM::CallbacksMap buildMSICallbacks(unsigned oemIndex,
                                                   const Us4OEMInterruptCallbacksMap &callbacks) {
        IUs4OEM::CallbacksMap result;
        if(callbacks.empty()) {
            return result;
        }
        const Ordinal oem = static_cast<Ordinal>(oemIndex);
        for(const auto &[apiIrq, callback] : callbacks) {
            if(!callback) {
                continue;
            }
            result[toMSINumber(apiIrq)] = [callback, oem]() {
                callback(oem);
            };
        }
        return result;
    }

    static IUs4OEM::MSINumber toMSINumber(Us4OEMInterrupt interrupt) {
        switch(interrupt) {
            case Us4OEMInterrupt::PROBE_NOT_CONNECTED: return IUs4OEM::MSINumber::PROBE_NOT_CONNECTED;
            case Us4OEMInterrupt::PULSER_INTERRUPT:    return IUs4OEM::MSINumber::PULSERINTERRUPT;
            case Us4OEMInterrupt::TX_TIMEOUT:          return IUs4OEM::MSINumber::TX_TIMEOUT;
            case Us4OEMInterrupt::WATCHDOG_IRQ0:       return IUs4OEM::MSINumber::WATCHDOG_IRQ0;
            case Us4OEMInterrupt::WATCHDOG_IRQ1:       return IUs4OEM::MSINumber::WATCHDOG_IRQ1;
            case Us4OEMInterrupt::HVPS_FUSE:           return IUs4OEM::MSINumber::HVPS_FUSE;
        }
        throw ::arrus::IllegalArgumentException("Unknown Us4OEMInterrupt value");
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_EXTERNAL_IUS4OEM_IUS4OEMFACTORYIMPL_H
