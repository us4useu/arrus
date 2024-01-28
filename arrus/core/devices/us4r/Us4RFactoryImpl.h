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
#include "arrus/core/devices/us4r/hv/HighVoltageSupplierFactory.h"
#include "arrus/core/devices/us4r/backplane/DigitalBackplaneFactory.h"
#include "arrus/core/devices/us4r/Us4RSettingsConverter.h"

namespace arrus::devices {

class Us4RFactoryImpl : public Us4RFactory {
 public:
    Us4RFactoryImpl(std::unique_ptr<Us4OEMFactory> us4oemFactory,
                    std::unique_ptr<ProbeAdapterFactory> adapterFactory,
                    std::unique_ptr<ProbeFactory> probeFactory,
                    std::unique_ptr<IUs4OEMFactory> ius4oemFactory,
                    std::unique_ptr<IUs4OEMInitializer> ius4oemInitializer,
                    std::unique_ptr<Us4RSettingsConverter> us4RSettingsConverter,
                    std::unique_ptr<HighVoltageSupplierFactory> hvFactory,
                    std::unique_ptr<DigitalBackplaneFactory> backplaneFactory)
        : ius4oemFactory(std::move(ius4oemFactory)),
          ius4oemInitializer(std::move(ius4oemInitializer)),
          us4oemFactory(std::move(us4oemFactory)),
          us4RSettingsConverter(std::move(us4RSettingsConverter)),
          probeAdapterFactory(std::move(adapterFactory)),
          probeFactory(std::move(probeFactory)),
          hvFactory(std::move(hvFactory)),
          backplaneFactory(std::move(backplaneFactory)){}

    Us4R::Handle getUs4R(Ordinal ordinal, const Us4RSettings &settings) override {
        DeviceId id(DeviceType::Us4R, ordinal);

        // Validate us4r settings (general).
        // TODO validate nus4oems and mapping
        Us4RSettingsValidator validator(ordinal);
        validator.validate(settings);
        validator.throwOnErrors();

        if (settings.getProbeAdapterSettings().has_value()) {
            // Probe, Adapter -> Us4OEM settings.
            // Adapter
            auto &probeAdapterSettings = settings.getProbeAdapterSettings().value();
            ProbeAdapterSettingsValidator adapterValidator(0);
            adapterValidator.validate(probeAdapterSettings);
            adapterValidator.throwOnErrors();
            // Probe
            auto probeSettings = settings.getProbeSettings().value();
            // TODO validate probe settings
            auto &rxSettings = settings.getRxSettings().value();
            // Rx settings will be validated by a specific device
            // (Us4OEMs validator)
            // Convert to Us4OEM settings
            auto[us4OEMSettings, adapterSettings] = us4RSettingsConverter->convertToUs4OEMSettings(
                probeAdapterSettings, probeSettings, rxSettings,
                settings.getChannelsMask(),
                settings.getReprogrammingMode(),
                settings.getNumberOfUs4oems(),
                settings.getAdapterToUs4RModuleNumber(),
                settings.getTxFrequencyRange()
            );

            // verify if the generated us4oemSettings.channelsMask is equal to us4oemChannelsMask field
            validateChannelsMasks(us4OEMSettings, settings.getUs4OEMChannelsMask());

            auto[us4oems, masterIUs4OEM] =
                getUs4OEMs(us4OEMSettings, settings.isExternalTrigger(), probeAdapterSettings.getIOSettings());
            std::vector<Us4OEMImplBase::RawHandle> us4oemPtrs(us4oems.size());
            std::transform(std::begin(us4oems), std::end(us4oems), std::begin(us4oemPtrs),
                [](const Us4OEMImplBase::Handle &ptr) { return ptr.get(); });
            // validate probe and adapter settings
            ProbeAdapterSettingsValidator validator(id.getOrdinal());
            validator.validate(adapterSettings);
            validator.throwOnErrors();
            probeSettings

            ProbeImplBase::Handle probe = probeFactory->getProbe(probeSettings, adapter.get());

            std::vector<IUs4OEM*> ius4oems;
            for(auto &us4oem: us4oems) {
                ius4oems.push_back(us4oem->getIUs4OEM());
            }

            auto backplane = getBackplane(settings.getDigitalBackplaneSettings(), settings.getHVSettings(), ius4oems);
            auto hv = getHV(settings.getHVSettings(), ius4oems, backplane);
            return std::make_unique<Us4RImpl>(id, std::move(us4oems), adapter, probe, std::move(hv), rxSettings,
                                              settings.getChannelsMask(), std::move(backplane),
                                              settings.getBitstreams(),
                                              probeSettings.getBitstreamId().has_value()
                                              );
        } else {
            throw IllegalArgumentException("Custom OEM configuration is no longer available.");
        }
    }

 private:

    void validateChannelsMasks(const std::vector<Us4OEMSettings> &us4oemSettings,
                               const std::vector<std::vector<uint8>> &us4oemChannelsMasks) {
        ARRUS_REQUIRES_TRUE_E(
            us4oemSettings.size() == us4oemChannelsMasks.size(),
            ::arrus::IllegalArgumentException(
                format("There should be exactly {} us4oem channels masks in the system configuration.",
                       us4oemSettings.size()))
        );
        for (unsigned i = 0; i < us4oemSettings.size(); ++i) {
            auto &setting = us4oemSettings[i]; // inferred from probe element masking
            std::unordered_set<uint8> us4oemMask(
                    std::begin(us4oemChannelsMasks[i]),
                    std::end(us4oemChannelsMasks[i])); // provided by user
            if(!setting.getChannelsMask().empty() || !us4oemMask.empty()) {
                if(us4oemMask.empty()) {
                    // Avoid additional validation (for convenience) when no us4OEM channels were explicitly provided.
                    getDefaultLogger()->log(LogSeverity::WARNING,
                                            format("No channel masking provided explicitly for us4OEM {}, "
                                            "I am skipping additional validation.", i));
                }
                else {
                    ARRUS_REQUIRES_TRUE_E(
                        setting.getChannelsMask() == us4oemMask,
                        ::arrus::IllegalArgumentException(
                            format(
                            "The provided us4r channels masks does not match the provided us4oem channels masks, "
                            "for us4oem {}", i)));
                }
            }
        }
    }

    /**
     * @return a pair: us4oems, master ius4oem
     */
    std::pair<std::vector<Us4OEMImplBase::Handle>, IUs4OEM *>
    getUs4OEMs(const std::vector<Us4OEMSettings> &us4oemCfgs, bool isExternalTrigger, const us4r::IOSettings& io) {
        ARRUS_REQUIRES_AT_LEAST(us4oemCfgs.size(), 1,"At least one us4oem should be configured.");
        auto nUs4oems = static_cast<Ordinal>(us4oemCfgs.size());

        // Initialize Us4OEMs.
        // We need to initialize Us4OEMs on a Us4R system level.
        // This is because Us4OEM initialization procedure needs to consider
        // existence of some master module (by default it's the 'Us4OEM:0').
        // Check the initializeModules function to see why.
        std::vector<IUs4OEMHandle> ius4oems = ius4oemFactory->getModules(nUs4oems);

        // Modifies input list - sorts ius4oems by ID in ascending order.
        ius4oemInitializer->sortModulesById(ius4oems);

        // Pre-configure us4oems.
        for(size_t i = 0; i < us4oemCfgs.size(); ++i) {
            ius4oems[i]->SetTxFrequencyRange(us4oemCfgs[i].getTxFrequencyRange());
        }

        ius4oemInitializer->initModules(ius4oems);
        auto master = ius4oems[0].get();

        // Create Us4OEMs.
        Us4RImpl::Us4OEMs us4oems;
        ARRUS_REQUIRES_EQUAL(ius4oems.size(), us4oemCfgs.size(),
                             ArrusException("Values are not equal: ius4oem size, us4oem settings size"));

        // Determine which OEMs must acquire RX nops, for pulse counter capability.
        std::unordered_set<Ordinal> pulseCounterOems;
        if(io.hasFrameMetadataCapability())  {
            pulseCounterOems = io.getFrameMetadataCapabilityOEMs();
        }
        else {
            // By default us4OEM:0 is the pulse counter.
            pulseCounterOems.insert(Ordinal(0));
        }
        for (unsigned i = 0; i < ius4oems.size(); ++i) {
            // TODO(Us4R-10) use ius4oem->GetDeviceID() as an ordinal number, instead of value of i
            auto ordinal = static_cast<Ordinal>(i);
            us4oems.push_back(us4oemFactory->getUs4OEM(
                static_cast<Ordinal>(i),
                ius4oems[i],
                us4oemCfgs[i],
                isExternalTrigger,
                setContains(pulseCounterOems, ordinal) // accept RX nops?
                // NOTE: the above should be consistent with the ProbeAdapterImpl::frameMetadataOem
            ));
        }
        initCapabilities(us4oems, io);
        return {std::move(us4oems), master};
    }

    std::vector<HighVoltageSupplier::Handle> getHV(const std::optional<HVSettings> &settings,
                                                   std::vector<IUs4OEM *> &us4oems,
                                                   const std::optional<DigitalBackplane::Handle> &backplane) {
        if (settings.has_value()) {
            const auto &hvSettings = settings.value();
            return hvFactory->getHighVoltageSupplier(hvSettings, us4oems, backplane);
        } else {
            std::vector<HighVoltageSupplier::Handle> empty;
            return empty;
        }
    }

    std::optional<DigitalBackplane::Handle> getBackplane(
        const std::optional<DigitalBackplaneSettings> &dbarSettings,
        const std::optional<HVSettings> &hvSettings,
        std::vector<IUs4OEM *> &us4oems) {

        if(dbarSettings.has_value()) {
            return backplaneFactory->getDigitalBackplane(dbarSettings.value(), us4oems);
        }
        else if (hvSettings.has_value()) {
            // Fallback option: try to determine HV model based on the HV in use.
            return backplaneFactory->getDigitalBackplane(hvSettings.value(), us4oems);
        } else {
            return std::nullopt;
        }
    }

    void initCapabilities(const Us4RImpl::Us4OEMs& us4oems, const us4r::IOSettings settings) {
        if(settings.hasProbeConnectedCheckCapability()) {
            auto addr = settings.getProbeConnectedCheckCapabilityAddress();
            if(addr.getUs4OEM() == 0){
                us4oems.at(addr.getUs4OEM())->getIUs4OEM()->EnableProbeCheck(addr.getIO());
            }
            else {
                throw arrus::IllegalArgumentException("Probe check functionality must be connected to us4OEM #0");
            }
        }
    }

    std::unique_ptr<IUs4OEMFactory> ius4oemFactory;
    std::unique_ptr<IUs4OEMInitializer> ius4oemInitializer;
    std::unique_ptr<Us4OEMFactory> us4oemFactory;
    std::unique_ptr<Us4RSettingsConverter> us4RSettingsConverter;
    std::unique_ptr<ProbeAdapterFactory> probeAdapterFactory;
    std::unique_ptr<ProbeFactory> probeFactory;
    std::unique_ptr<HighVoltageSupplierFactory> hvFactory;
    std::unique_ptr<DigitalBackplaneFactory> backplaneFactory;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4RFACTORYIMPL_H