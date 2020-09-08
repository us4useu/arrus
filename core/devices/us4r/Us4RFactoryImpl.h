#ifndef ARRUS_CORE_DEVICES_US4R_US4RFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4RFACTORYIMPL_H

#include <numeric>
#include <stdexcept>
#include <boost/range/combine.hpp>


#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMInitializer.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterFactory.h"
#include "arrus/core/devices/probe/ProbeFactory.h"
#include "arrus/common/asserts.h"
#include "arrus/core/devices/us4r/Us4RFactory.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/us4r/Us4RImpl.h"
#include "arrus/core/devices/us4r/Us4RSettingsValidator.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterSettingsValidator.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMFactory.h"

#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"
#include "arrus/core/devices/us4r/Us4RSettingsConverter.h"

namespace arrus {

class Us4RFactoryImpl : public Us4RFactory {
public:
    Us4RFactoryImpl(std::unique_ptr<Us4OEMFactory> us4oemFactory,
                    std::unique_ptr<ProbeAdapterFactory> adapterFactory,
                    std::unique_ptr<ProbeFactory> probeFactory,
                    std::unique_ptr<IUs4OEMFactory> ius4oemFactory,
                    std::unique_ptr<IUs4OEMInitializer> ius4oemInitializer,
                    std::unique_ptr<Us4RSettingsConverter> us4RSettingsConverter)
        : ius4oemFactory(std::move(ius4oemFactory)),
          ius4oemInitializer(std::move(ius4oemInitializer)),
          us4oemFactory(std::move(us4oemFactory)),
          us4RSettingsConverter(std::move(us4RSettingsConverter)),
          probeAdapterFactory(std::move(adapterFactory)),
          probeFactory(std::move(probeFactory)) {}


    Us4R::Handle
    getUs4R(Ordinal ordinal, const Us4RSettings &settings) override {
        DeviceId id(DeviceType::Us4R, ordinal);

        // Validate us4r settings (general).
        Us4RSettingsValidator validator(ordinal);
        validator.validate(settings);
        validator.throwOnErrors();

        if(settings.getProbeAdapterSettings().has_value()) {
            // Probe, Adapter -> Us4OEM settings.
            // Adapter
            auto &probeAdapterSettings =
                settings.getProbeAdapterSettings().value();
            ProbeAdapterSettingsValidator adapterValidator(0);
            adapterValidator.validate(probeAdapterSettings);
            adapterValidator.throwOnErrors();
            // Probe
            auto &probeSettings =
                settings.getProbeSettings().value();
            // TODO validate probe settings
            auto &rxSettings =
                settings.getRxSettings().value();
            // Rx settings will be validated by a specific device
            // (Us4OEMs validator)

            // Convert to Us4OEM settings
            auto[us4OEMSettings, adapterSettings] =
            us4RSettingsConverter->convertToUs4OEMSettings(
                probeAdapterSettings, probeSettings, rxSettings);

            std::vector<Us4OEM::Handle> us4oems = getUs4OEMs(us4OEMSettings);
            std::vector<Us4OEM::RawHandle> us4oemPtrs(us4oems.size());
            std::transform(
                std::begin(us4oems), std::end(us4oems),
                std::begin(us4oemPtrs),
                [](const Us4OEM::Handle &ptr) { return ptr.get(); });
            // Create adapter.
            ProbeAdapter::Handle adapter =
                probeAdapterFactory->getProbeAdapter(adapterSettings,
                                                     us4oemPtrs);
            // Create probe.
            Probe::Handle probe = probeFactory->getProbe(probeSettings,
                                                         adapter.get());
            return std::make_unique<Us4RImpl>(id, us4oems, adapter, probe);
        } else {
            // Custom Us4OEMs only
            std::vector<Us4OEM::Handle> us4oems = getUs4OEMs(
                settings.getUs4OEMSettings());
            return std::make_unique<Us4RImpl>(id, us4oems);
        }
    }

private:
    std::vector<Us4OEM::Handle>
    getUs4OEMs(const std::vector<Us4OEMSettings> &us4oemCfgs) {
        ARRUS_REQUIRES_AT_LEAST(us4oemCfgs.size(), 1,
                                "At least one us4oem should be configured.");
        Ordinal nUs4oems = static_cast<Ordinal>(us4oemCfgs.size());

        // Initialize Us4OEMs.
        // We need to initialize Us4OEMs on a Us4R system level.
        // This is because Us4OEM initialization procedure needs to consider
        // existence of some master module (by default it's the 'Us4OEM:0').
        // Check the initializeModules function to see why.
        std::vector<IUs4OEMHandle> ius4oems =
            ius4oemFactory->getModules(nUs4oems);

        ius4oemInitializer->initModules(ius4oems);

        // Create Us4OEMs.
        Us4RImpl::Us4OEMs us4oems;
        ARRUS_REQUIRES_EQUAL(ius4oems.size(), us4oemCfgs.size(),
                             ArrusException(
                                 "Values are not equal: ius4oem size, "
                                 "us4oem settings size"));

        for(unsigned i = 0; i < ius4oems.size(); ++i) {
            us4oems.push_back(
                us4oemFactory->getUs4OEM(
                    static_cast<ChannelIdx>(i),
                    ius4oems[i], us4oemCfgs[i])
            );
        }
        return us4oems;
    }

    std::unique_ptr<IUs4OEMFactory> ius4oemFactory;
    std::unique_ptr<IUs4OEMInitializer> ius4oemInitializer;
    std::unique_ptr<Us4OEMFactory> us4oemFactory;
    std::unique_ptr<Us4RSettingsConverter> us4RSettingsConverter;
    std::unique_ptr<ProbeAdapterFactory> probeAdapterFactory;
    std::unique_ptr<ProbeFactory> probeFactory;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4RFACTORYIMPL_H
