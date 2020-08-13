#ifndef ARRUS_CORE_DEVICES_US4R_US4RFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4RFACTORYIMPL_H

#include <numeric>
#include <stdexcept>
#include <boost/range/combine.hpp>

#include "arrus/core/common/asserts.h"

#include "arrus/core/devices/us4r/Us4RFactory.h"
#include "arrus/core/devices/us4r/Us4RImpl.h"
#include "arrus/core/devices/us4r/Us4RSettingsValidator.h"
#include "arrus/core/devices/us4r/Us4RSettingsConverter.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMFactory.h"

#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"

namespace arrus {

class Us4RFactoryImpl : public Us4RFactory {
public:
    Us4RFactoryImpl(Us4OEMFactory &us4oemFactory,
                    IUs4OEMFactory &ius4oemFactory)
            : us4oemFactory(us4oemFactory), ius4oemFactory(ius4oemFactory) {}

    Us4R::Handle
    getUs4R(Ordinal ordinal, const Us4RSettings &settings) override {
        DeviceId id(DeviceType::Us4R, ordinal);

        // Validate us4r settings (general).
        Us4RSettingsValidator validator(ordinal);
        validator.validate(settings);
        validator.throwOnErrors();

        if(settings.getProbeAdapterSettings().has_value()) {
            // Probe Adapter
            auto &probeAdapterSettings =
                    settings.getProbeAdapterSettings().value();
            auto &rxSettings =
                    settings.getRxSettings().value();

            std::vector<Us4OEMSettings> us4OEMSettings =
                    Us4RSettingsConverter::convertToUs4OEMSettings(
                            probeAdapterSettings, rxSettings);
            std::vector<Us4OEM::Handle> us4oems = getUs4OEMs(
                    us4OEMSettings);

            // Probe
            if(settings.getProbeSettings().has_value()) {
                // TODO(pjarosik) implement probe settings
                throw std::runtime_error("NYI");
            } else {

            }
        } else {
            // Custom Us4OEMs only
            std::vector<Us4OEM::Handle> us4oems = getUs4OEMs(
                    settings.getUs4OEMSettings());
            return Us4R::Handle(new Us4RImpl(id, std::move(us4oems)));
        }
    }

private:

    std::vector<Us4OEM::Handle>
    getUs4OEMs(const std::vector<Us4OEMSettings> &us4oemCfgs) {
        ARRUS_REQUIRES_AT_LEAST(us4oemCfgs.size(), 1,
                                "At least one us4oem should be configured.");
        Ordinal nUs4oems = us4oemCfgs.size();

        // Initialize Us4OEMs.
        // We need to initialize Us4OEMs on a Us4R system level.
        // This is because Us4OEM initialization procedure needs to consider
        // existence of some master module (by default it's the 'Us4OEM:0').
        // Check the initializeModules function to see why.
        std::vector<IUs4OEMHandle> ius4oems = initializeModules(nUs4oems);

        // Create Us4OEMs.
        Us4RImpl::Us4OEMs us4oems;
        ARRUS_REQUIRES_EQUAL(ius4oems.size(), us4oemCfgs.size(),
                             ArrusException(
                                     "Values are not equal: ius4oem size, "
                                     "us4oem settings size"));

        for(unsigned i = 0; i < us4oems.size(); ++i) {
            us4oems.push_back(
                    us4oemFactory.getUs4OEM(i, ius4oems[i], us4oemCfgs[i])
            );
        }
        return us4oems;
    }

    /**
     * Creates IUs4OEM handles and initializes them according to
     */
    std::vector<IUs4OEMHandle> initializeModules(const Ordinal nModules) {
        std::vector<IUs4OEMHandle> us4oems;

        std::vector<Ordinal> ordinals(nModules);
        std::iota(std::begin(ordinals), std::end(ordinals), 0);

        // Create Us4OEM handles.
        for(auto ordinal : ordinals) {
            us4oems.push_back(ius4oemFactory.getIUs4OEM(ordinal));
        }
        // Reorder us4oems according to ids (us4oem with the lowest id is the
        // first one, with the highest id - the last one).
        // TODO(pjarosik) make the below sorting exception safe
        // (currently will std::terminate on an exception).
        std::sort(std::begin(us4oems), std::end(us4oems),
                  [](const IUs4OEMHandle &x, const IUs4OEMHandle &y) {
                      return x->GetID() < y->GetID();
                  });

        for(auto &u : us4oems) {
            u->Initialize(1);
        }
        // Perform successive initialization levels.
        for(int level = 2; level <= 4; level++) {
            us4oems[0]->Synchronize();
            for(auto &u : us4oems) {
                u->Initialize(level);
            }
        }
        // Us4OEMs are initialized here.
        return us4oems;
    }


    Us4OEMFactory &us4oemFactory;
    IUs4OEMFactory &ius4oemFactory;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4RFACTORYIMPL_H
