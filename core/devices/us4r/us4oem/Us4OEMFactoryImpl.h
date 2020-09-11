#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMFACTORYIMPL_H

#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/common/asserts.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMFactory.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMSettingsValidator.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"
#include "arrus/core/devices/us4r/external/ius4oem/PGAGainValueMap.h"

namespace arrus::devices {
class Us4OEMFactoryImpl : public Us4OEMFactory {
public:
    Us4OEMFactoryImpl() = default;

    Us4OEM::Handle
    getUs4OEM(Ordinal ordinal, IUs4OEMHandle &ius4oem,
              const Us4OEMSettings &cfg) override {

        // Validate settings.
        Us4OEMSettingsValidator validator(ordinal);
        validator.throwOnErrors();

        // We assume here, that the ius4oem is already initialized.

        // Configure IUs4OEM
        ChannelIdx chGroupSize = Us4OEMImpl::N_RX_CHANNELS;
        ARRUS_REQUIRES_TRUE(
            IUs4OEM::NCH % chGroupSize == 0,
            arrus::format("Number of Us4OEM channels ({}) is not "
                          "divisible by the size of channel group ({})",
                          IUs4OEM::NCH, chGroupSize));
        ChannelIdx nChannelGroups = IUs4OEM::NCH / chGroupSize;

        // Tx channel mapping
        // Convert to uint8_t
        std::vector<uint8_t> channelMapping;
        ARRUS_REQUIRES_AT_MOST(
            cfg.getChannelMapping().size(),
            UINT8_MAX,
            arrus::format("Maximum number of channels: {}", UINT8_MAX));

        for(auto value : cfg.getChannelMapping()) {
            ARRUS_REQUIRES_AT_MOST(value, (ChannelIdx) UINT8_MAX, arrus::format(
                "Us4OEM channel index cannot exceed {}",
                (ChannelIdx) UINT8_MAX));
            channelMapping.push_back(static_cast<uint8_t>(value));
        }

        uint8_t virtualIdx = 0;
        for(uint8_t physicalIdx : channelMapping) {
            // src - physical channel
            // dst - virtual channel
            ius4oem->SetTxChannelMapping(virtualIdx++, physicalIdx);
        }
        // Rx channel mapping
        // Check if the the permutation in channel mapping is the same
        // for each group of 32 channels. If so, use the first 32 channels
        // to set mapping.
        // Otherwise store rxChannelMapping in Us4OEM handle for further usage.

        const bool isSinglePermutation = hasConsistentPermutations(
            cfg.getChannelMapping(), chGroupSize, nChannelGroups);

        if(isSinglePermutation) {
            ius4oem->SetRxChannelMapping(
                std::vector<uint8_t>(
                    std::begin(channelMapping),
                    std::begin(channelMapping) + chGroupSize),
                0);
        }
        // otherwise store the complete channel mapping array in Us4OEM handle
        // (check the value returned by current method).

        // Other parameters
        // TGC
        const auto pgaGain = cfg.getRxSettings().getPGAGain();
        const auto lnaGain = cfg.getRxSettings().getLNAGain();
        ius4oem->SetPGAGain(
            PGAGainValueMap::getInstance().getEnumValue(pgaGain));
        ius4oem->SetLNAGain(
            LNAGainValueMap::getInstance().getEnumValue(lnaGain));
        // Convert TGC values to [0, 1] range
        if(cfg.getRxSettings().getTGCSamples().empty()) {
            ius4oem->TGCDisable();
        } else {
            const auto maxGain = pgaGain + lnaGain;
            // TODO(pjarosik) extract a common function to compute normalized tgc samples
            const RxSettings::TGCCurve normalizedTGCSamples = getNormalizedTGCSamples(
                cfg.getRxSettings().getTGCSamples(),
                maxGain - Us4OEMImpl::TGC_ATTENUATION_RANGE,
                static_cast<RxSettings::TGCSample>(maxGain));

            ius4oem->TGCEnable();
            // Currently firing parameter does not matter.
            ius4oem->TGCSetSamples(normalizedTGCSamples, 0);
        }

        // DTGC
        if(cfg.getRxSettings().getDTGCAttenuation().has_value()) {
            ius4oem->SetDTGC(us4r::afe58jd18::EN_DIG_TGC::EN_DIG_TGC_EN,
                             DTGCAttenuationValueMap::getInstance().getEnumValue(
                                 cfg.getRxSettings().getDTGCAttenuation().value()));
        } else {
            // DTGC value does not matter
            ius4oem->SetDTGC(us4r::afe58jd18::EN_DIG_TGC::EN_DIG_TGC_DIS,
                             us4r::afe58jd18::DIG_TGC_ATTENUATION::
                             DIG_TGC_ATTENUATION_42dB);
        }

        // Filtering
        ius4oem->SetLPFCutoff(LPFCutoffValueMap::getInstance().getEnumValue(
            cfg.getRxSettings().getLPFCutoff()));

        // Active termination
        if(cfg.getRxSettings().getActiveTermination().has_value()) {
            ius4oem->SetActiveTermination(
                us4r::afe58jd18::ACTIVE_TERM_EN::ACTIVE_TERM_EN,
                ActiveTerminationValueMap::getInstance().getEnumValue(
                    cfg.getRxSettings().getActiveTermination().value()));
        } else {
            ius4oem->SetActiveTermination(
                us4r::afe58jd18::ACTIVE_TERM_EN::ACTIVE_TERM_DIS,
                us4r::afe58jd18::GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_50);
        }


        return std::make_unique<Us4OEMImpl>(
            DeviceId(DeviceType::Us4OEM, ordinal),
            std::move(ius4oem), cfg.getActiveChannelGroups(),
            channelMapping);
    }

private:

    static RxSettings::TGCCurve
    getNormalizedTGCSamples(const RxSettings::TGCCurve &samples,
                            const RxSettings::TGCSample min,
                            const RxSettings::TGCSample max) {
        RxSettings::TGCCurve result;
        auto range = max - min;
        std::transform(std::begin(samples), std::end(samples),
                       std::back_inserter(result),
                       [=](auto value) { return (value - min) / range; });
        return result;
    }

    static bool
    hasConsistentPermutations(const std::vector<ChannelIdx> &vector,
                              ChannelIdx groupSize,
                              ChannelIdx nGroups) {
        for(ChannelIdx group = 1; group < nGroups; ++group) {
            for(ChannelIdx i = 0; i < groupSize; ++i) {
                if(vector[i] != (vector[i + group * groupSize] % groupSize)) {
                    return false;
                }
            }
        }
        return true;
    }
};
}

#endif //ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMFACTORYIMPL_H
