#ifndef ARRUS_CORE_DEVICES_US4R_US4RFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4RFACTORYIMPL_H

#include <stdexcept>
#include <unordered_set>
#include <algorithm>
#include <iterator>

#include "arrus/common/asserts.h"
#include "arrus/core/devices/probe/ProbeSettingsValidator.h"
#include "arrus/core/devices/us4r/Us4RFactory.h"
#include "arrus/core/devices/us4r/Us4RImpl.h"
#include "arrus/core/devices/us4r/Us4RSettingsValidator.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMInitializer.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterSettingsValidator.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMFactory.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"

#include "arrus/core/devices/us4r/Us4RSettingsConverter.h"
#include "arrus/core/devices/us4r/backplane/DigitalBackplaneFactory.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"
#include "arrus/core/devices/us4r/hv/HighVoltageSupplierFactory.h"

namespace arrus::devices {

class Us4RFactoryImpl : public Us4RFactory {
public:
    Us4RFactoryImpl(std::unique_ptr<Us4OEMFactory> us4oemFactory,
                    std::unique_ptr<IUs4OEMFactory> ius4oemFactory,
                    std::unique_ptr<IUs4OEMInitializer> ius4oemInitializer,
                    std::unique_ptr<Us4RSettingsConverter> us4RSettingsConverter,
                    std::unique_ptr<HighVoltageSupplierFactory> hvFactory,
                    std::unique_ptr<DigitalBackplaneFactory> backplaneFactory)
        : ius4oemFactory(std::move(ius4oemFactory)), ius4oemInitializer(std::move(ius4oemInitializer)),
          us4oemFactory(std::move(us4oemFactory)), us4RSettingsConverter(std::move(us4RSettingsConverter)),
          hvFactory(std::move(hvFactory)), backplaneFactory(std::move(backplaneFactory)) {}

    Us4R::Handle getUs4R(Ordinal ordinal, const Us4RSettings &settings) override {
        DeviceId id(DeviceType::Us4R, ordinal);

        // Validate us4r settings (general).
        // TODO validate nus4oems and mapping
        Us4RSettingsValidator validator(ordinal);
        validator.validate(settings);
        validator.throwOnErrors();

        WatchdogSettingsValidator watchdogValidator;
        watchdogValidator.validate(settings.getWatchdogSettings());
        watchdogValidator.throwOnErrors();

        if (settings.getProbeAdapterSettings().has_value()) {
            // Probe, Adapter -> Us4OEM settings.
            // Adapter
            auto &probeAdapterSettings = settings.getProbeAdapterSettings().value();
            ProbeAdapterSettingsValidator adapterValidator(0);
            adapterValidator.validate(probeAdapterSettings);
            adapterValidator.throwOnErrors();
            // Probes list
            auto probeSettings = settings.getProbeSettingsList();
            // TODO validate probe settings
            auto &rxSettings = settings.getRxSettings().value();
            // Rx settings will be validated by a specific device
            // (Us4OEMs validator)
            // Convert to Us4OEM settings
            auto [us4OEMSettings, adapterSettings] = us4RSettingsConverter->convertToUs4OEMSettings(
                probeAdapterSettings, rxSettings, settings.getReprogrammingMode(), settings.getNumberOfUs4oems(),
                settings.getAdapterToUs4RModuleNumber(), settings.getTxFrequencyRange());

            // Get IUs4OEM handles only, without initializing them.
            // This is in order to enable internal trigger before
            // OEMs are initialized.
            auto ius4oemHandles = getIUS4OEMs((Ordinal)(us4OEMSettings.size()), settings.isAllowDuplicateOEMIds());

            // initialization
            std::vector<IUs4OEM*> ius4oems;
            for(auto &handle: ius4oemHandles) {
                ius4oems.push_back(handle.get());
            }
            auto backplane = getBackplane(settings.getDigitalBackplaneSettings(), settings.getHVSettings(), ius4oems);
            if(backplane.has_value()) {
                backplane.value()->enableInternalTrigger();
            }

            auto [us4oems, masterIUs4OEM] =
                getUs4OEMs(us4OEMSettings, settings.isExternalTrigger(), probeAdapterSettings.getIOSettings(),
                           ius4oemHandles, settings.getTxRxLimits(), settings.getWatchdogSettings());
            std::vector<Us4OEMImplBase::RawHandle> us4oemPtrs(us4oems.size());
            std::transform(std::begin(us4oems), std::end(us4oems), std::begin(us4oemPtrs),
                           [](const Us4OEMImplBase::Handle &ptr) { return ptr.get(); });
            // validate probe and adapter settings
            ProbeAdapterSettingsValidator paValidator(id.getOrdinal());
            paValidator.validate(adapterSettings);
            paValidator.throwOnErrors();
            Ordinal probeOrdinal = 0;
            bool isBitstreamAddr = isBitstreamAddressing(probeSettings);
            for (const auto &s : probeSettings) {
                DeviceId probeId(DeviceType::Probe, probeOrdinal);
                ProbeSettingsValidator pValidator(probeId.getOrdinal());
                pValidator.validate(s);
                pValidator.throwOnErrors();
                probeOrdinal++;
            }
            bool isExternalTrigger = false;
            if(backplane.has_value() && settings.isExternalTrigger()) {
                isExternalTrigger = true;
            }
            auto hv = getHV(settings.getHVSettings(), ius4oems, backplane);
            return std::make_unique<Us4RImpl>(
                id, std::move(us4oems), std::move(probeSettings), std::move(adapterSettings), std::move(hv), rxSettings,
                settings.getChannelsMaskForAllProbes(), std::move(backplane), settings.getBitstreams(),
                isBitstreamAddr, adapterSettings.getIOSettings(), isExternalTrigger);
        } else {
            throw IllegalArgumentException("Custom OEM configuration is not available since 0.11.0.");
        }
    }

private:
    std::vector<IUs4OEMHandle> getIUS4OEMs(Ordinal nOEMs, bool allowDuplicateIds) {
        ARRUS_REQUIRES_AT_LEAST(nOEMs, 1, "At least one us4oem should be configured.");
        std::vector<IUs4OEMHandle> ius4oems = ius4oemFactory->getModules(nOEMs);
        // OEM initial validation
        validateOEMIds(ius4oems, allowDuplicateIds);
        // Modifies input list - sorts ius4oems by ID in ascending order.
        ius4oemInitializer->sortModulesById(ius4oems);
        return ius4oems;
    }

    /**
     * @return a pair: us4oems, master ius4oem
     */
    std::pair<std::vector<Us4OEMImplBase::Handle>, IUs4OEM *>
    getUs4OEMs(const std::vector<Us4OEMSettings> &us4oemCfgs, bool isExternalTrigger, const us4r::IOSettings &io,
               std::vector<IUs4OEMHandle> &ius4oems, const std::optional<Us4RTxRxLimits> &limits,
               const WatchdogSettings watchdogSettings) {
        // Pre-configure us4oems.
        for(size_t i = 0; i < us4oemCfgs.size(); ++i) {
            const auto &ius4oem = ius4oems.at(i);
            ius4oem->SetTxFrequencyRange(us4oemCfgs[i].getTxFrequencyRange());
            if(Us4OEMImpl::isOEMPlus(ius4oem->GetOemVersion())) {
                if(watchdogSettings.isEnabled()) {

                    ius4oem->SetOEMWatchdogThresholds(
                        // Convert to ms
                        ARRUS_SAFE_CAST(std::roundf(watchdogSettings.getOEMThreshold0()*1000), int16),
                        ARRUS_SAFE_CAST(std::roundf(watchdogSettings.getOEMThreshold1()*1000), int16)
                    );
                    ius4oem->SetHostWatchdogThresholds(
                        // Convert to ms
                        ARRUS_SAFE_CAST(std::roundf(watchdogSettings.getHostThreshold()*1000), int16)
                    );
                }
                else {
                    ius4oem->DisableWatchdog();
                }
            }
        }
        // Initialize Us4OEMs.
        // We need to initialize Us4OEMs on a Us4R system level.
        // This is because Us4OEM initialization procedure needs to consider
        // existence of some master module (by default it's the 'Us4OEM:0').
        // Check the initializeModules function to see why.
        ius4oemInitializer->initModules(ius4oems);
        auto master = ius4oems[0].get();

        // Create Us4OEMs.
        Us4RImpl::Us4OEMs us4oems;
        ARRUS_REQUIRES_EQUAL(ius4oems.size(), us4oemCfgs.size(),
                             ArrusException("Values are not equal: ius4oem size, us4oem settings size"));

        // Determine which OEMs must acquire RX nops, for pulse counter capability.
        std::unordered_set<Ordinal> pulseCounterOems;
        if (io.hasFrameMetadataCapability()) {
            pulseCounterOems = io.getFrameMetadataCapabilityOEMs();
        }
        for (unsigned i = 0; i < ius4oems.size(); ++i) {
            // TODO(Us4R-10) use ius4oem->GetDeviceID() as an ordinal number, instead of value of i
            auto ordinal = static_cast<Ordinal>(i);
            us4oems.push_back(us4oemFactory->getUs4OEM(
                static_cast<Ordinal>(i), ius4oems[i], us4oemCfgs[i], isExternalTrigger,
                setContains(pulseCounterOems, ordinal), // accept RX nops?
                // NOTE: the above should be consistent with the ProbeAdapterImpl::frameMetadataOem
                limits));
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

    std::optional<DigitalBackplane::Handle> getBackplane(const std::optional<DigitalBackplaneSettings> &dbarSettings,
                                                         const std::optional<HVSettings> &hvSettings,
                                                         std::vector<IUs4OEM *> &us4oems) {

        if (dbarSettings.has_value()) {
            return backplaneFactory->getDigitalBackplane(dbarSettings.value(), us4oems);
        } else if (hvSettings.has_value()) {
            // Fallback option: try to determine HV model based on the HV in use.
            return backplaneFactory->getDigitalBackplane(hvSettings.value(), us4oems);
        } else {
            return std::nullopt;
        }
    }

    void initCapabilities(const Us4RImpl::Us4OEMs &us4oems, const us4r::IOSettings settings) {
        if (settings.hasProbeConnectedCheckCapability()) {
            auto addr = settings.getProbeConnectedCheckCapabilityAddress();
            if (addr.getUs4OEM() == 0) {
                us4oems.at(addr.getUs4OEM())->getIUs4OEM()->EnableProbeCheck(addr.getIO());
            } else {
                throw arrus::IllegalArgumentException("Probe check functionality must be connected to us4OEM #0");
            }
        } else {
            for(auto &us4oem: us4oems) {
               us4oem->getIUs4OEM()->DisableProbeCheck();
            }
        }
    }

    bool isBitstreamAddressing(const std::vector<ProbeSettings> &probeSettings) {
        std::unordered_set<bool> f;
        std::transform(std::begin(probeSettings), std::end(probeSettings), std::inserter(f, std::begin(f)),
                       [](const auto &p){return p.getBitstreamId().has_value();});
        if(f.size() > 1) {
            throw IllegalArgumentException("All probes should be bitstream addressable or not.");
        }
        return *std::begin(f);
    }

    void validateOEMIds(const std::vector<IUs4OEMHandle> &oems, bool allowDuplicateIds) {
        std::unordered_set<unsigned int> idsSet;
        // transform vector to set
        std::transform(std::begin(oems), std::end(oems), std::inserter(idsSet, std::end(idsSet)), [](const auto &oem) {
            return oem->GetID();
        });

        if(idsSet.size() < oems.size()) {
            // We have some duplicates, prepare error/warning message.
            std::stringstream stringBuilder;
            stringBuilder << "Detected OEMs with non-unique IDs. ";
            for(size_t i = 0; i < oems.size(); ++i) {
                stringBuilder << "PCI device: " << i << ", ID: " << oems.at(i)->GetID()  << "; ";
            }
            if(allowDuplicateIds) {
                getDefaultLogger()->log(LogSeverity::WARNING, stringBuilder.str());
            }
            else {
                stringBuilder << " To replace this error with warning message, set `allow_duplicate_oem_ids` "
                                 "in the us4r settings to true.";
                throw IllegalArgumentException(stringBuilder.str());
            }
        }
    }

    std::unique_ptr<IUs4OEMFactory> ius4oemFactory;
    std::unique_ptr<IUs4OEMInitializer> ius4oemInitializer;
    std::unique_ptr<Us4OEMFactory> us4oemFactory;
    std::unique_ptr<Us4RSettingsConverter> us4RSettingsConverter;
    std::unique_ptr<HighVoltageSupplierFactory> hvFactory;
    std::unique_ptr<DigitalBackplaneFactory> backplaneFactory;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_US4RFACTORYIMPL_H
