#ifndef ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMFACTORYIMPL_H

#include "arrus/core/common/asserts.h"
#include "arrus/core/devices/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/us4oem/Us4OEMFactory.h"
#include "arrus/core/devices/us4oem/Us4OEMSettingsValidator.h"
#include "arrus/core/external/ius4oem/IUs4OEMFactory.h"
#include "arrus/core/external/ius4oem/PGAGainValueMap.h"

namespace arrus {
class Us4OEMFactoryImpl : public Us4OEMFactory {
public:
    explicit Us4OEMFactoryImpl(IUs4OEMFactory &ius4oemFactory)
            : ius4oemFactory(ius4oemFactory) {}


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
        ChannelIdx virtualIdx = 0;
        for(ChannelIdx physicalIdx : cfg.getChannelMapping()) {
            // src - physical channel
            // dst - virtual channel
            ius4oem->SetTxChannelMapping(virtualIdx, physicalIdx);
        }
        // Rx channel mapping
        // Check if the the permutation in channel mapping is the same
        // for each group of 32 channels. If so, use the first 32 channels
        // to set mapping.
        // Otherwise store rxChannelMapping in Us4OEM handle for further usage.

        const bool isStoreRxMapping = hasConsistentPermutations(
                cfg.getChannelMapping(), chGroupSize, nChannelGroups);

        if(!isStoreRxMapping) {
            ius4oem->SetRxChannelMapping(
                    std::vector<ChannelIdx>(
                            std::begin(cfg.getChannelMapping()),
                            std::begin(cfg.getChannelMapping()) + chGroupSize),
                    0);
        }
        // otherwise store the complete channel mapping array in Us4OEM handle
        // (check the value returned by current method).

        // Other parameters
        // TGC
        const auto pgaGain = cfg.getPGAGain();
        const auto lnaGain = cfg.getLNAGain();
        ius4oem->SetPGAGain(
                PGAGainValueMap::getInstance().getEnumValue(pgaGain));
        ius4oem->SetLNAGain(
                LNAGainValueMap::getInstance().getEnumValue(lnaGain));
        // Convert TGC values to [0, 1] range
        if(cfg.getTGCSamples().empty()) {
            ius4oem->TGCDisable();
        } else {
            const auto maxGain = pgaGain + lnaGain;
            // TODO(pjarosik) extract a common function to compute normalized tgc samples
            const TGCCurve normalizedTGCSamples = getNormalizedTGCSamples(
                    cfg.getTGCSamples(), maxGain - Us4OEMImpl::TGC_RANGE,
                    maxGain);

            ius4oem->TGCEnable();
            // Currently firing parameter does not matter.
            ius4oem->TGCSetSamples(normalizedTGCSamples, 0);
        }

        // DTGC
        if(cfg.getDTGCAttenuation().has_value()) {
            ius4oem->SetDTGC(us4r::afe58jd18::EN_DIG_TGC::EN_DIG_TGC_EN,
                             DTGCAttenuationValueMap::getInstance().getEnumValue(
                                     cfg.getDTGCAttenuation().value()));
        } else {
            // DTGC value does not matter
            ius4oem->SetDTGC(us4r::afe58jd18::EN_DIG_TGC::EN_DIG_TGC_DIS,
                             us4r::afe58jd18::DIG_TGC_ATTENUATION::
                             DIG_TGC_ATTENUATION_42dB);
        }

        // Filtering
        ius4oem->SetLPFCutoff(LPFCutoffValueMap::getInstance().getEnumValue(
                cfg.getLPFCutoff()));

        // Active termination
        if(cfg.getActiveTermination().has_value()) {
            ius4oem->SetActiveTermination(
                    us4r::afe58jd18::ACTIVE_TERM_EN::ACTIVE_TERM_EN,
                    ActiveTerminationValueMap::getInstance().getEnumValue(
                            cfg.getActiveTermination().value()));
        } else {
            ius4oem->SetActiveTermination(
                    us4r::afe58jd18::ACTIVE_TERM_EN::ACTIVE_TERM_DIS,
                    us4r::afe58jd18::GBL_ACTIVE_TERM::GBL_ACTIVE_TERM_50);
        }

        // Prepare mapping to store in the Us4OEM intance.
        std::vector<ChannelIdx> rxChannelMapping;
        if(isStoreRxMapping) {
            rxChannelMapping = cfg.getChannelMapping();
        } else {
            rxChannelMapping = {};
        }
        return Us4OEM::Handle(
                new Us4OEMImpl(
                        DeviceId(DeviceType::Us4OEM, ordinal),
                        std::move(ius4oem), cfg.getActiveChannelGroups(),
                        rxChannelMapping));
    }

private:

    static TGCCurve getNormalizedTGCSamples(const TGCCurve &samples,
                                            const TGCSampleValue min,
                                            const TGCSampleValue max) {
        TGCCurve result(samples.size());
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

    IUs4OEMFactory &ius4oemFactory;
};
}

#endif //ARRUS_CORE_DEVICES_US4OEM_IMPL_US4OEMFACTORYIMPL_H
