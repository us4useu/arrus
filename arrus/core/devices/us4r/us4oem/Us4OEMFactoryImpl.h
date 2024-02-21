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

    Us4OEMImplBase::Handle getUs4OEM(Ordinal ordinal, IUs4OEMHandle &ius4oem, const Us4OEMSettings &cfg,
                                     bool isExternalTrigger, bool acceptRxNops = false) override {
        // Validate settings.
        Us4OEMSettingsValidator validator(ordinal);
        validator.validate(cfg);
        validator.throwOnErrors();

        // We assume here, that the ius4oem is already initialized.
        // Configure IUs4OEM
        ChannelIdx chGroupSize = Us4OEMImpl::N_RX_CHANNELS;
        ARRUS_REQUIRES_TRUE(IUs4OEM::NCH % chGroupSize == 0,
                            arrus::format("Number of Us4OEM channels ({}) is not divisible by the size of channel group ({})",
                                    IUs4OEM::NCH, chGroupSize));
        ChannelIdx nChannelGroups = IUs4OEM::NCH / chGroupSize;

        // Tx channel mapping
        // Convert to uint8_t
        std::vector<uint8_t> channelMapping;
        ARRUS_REQUIRES_AT_MOST(cfg.getChannelMapping().size(), UINT8_MAX,
                               arrus::format("Maximum number of channels: {}", UINT8_MAX));

        for(auto value : cfg.getChannelMapping()) {
            ARRUS_REQUIRES_AT_MOST(value, (ChannelIdx) UINT8_MAX,
                                   arrus::format("Us4OEM channel index cannot exceed {}", (ChannelIdx) UINT8_MAX));
            channelMapping.push_back(static_cast<uint8_t>(value));
        }

        uint8_t virtualIdx = 0;
        for(uint8_t physicalIdx : channelMapping) {
            // src - physical channel
            // dst - virtual channel
            ius4oem->SetTxChannelMapping(physicalIdx, virtualIdx++);
        }
        // Rx channel mapping
        // Check if the the permutation in channel mapping is the same
        // for each group of 32 channels. If so, use the first 32 channels
        // to set mapping.
        // Otherwise store rxChannelMapping in Us4OEM handle for further usage.
        const bool isSinglePermutation = isConsistentPermutations(cfg.getChannelMapping(), chGroupSize, nChannelGroups);

        if(isSinglePermutation) {
            ius4oem->SetRxChannelMapping(
                    std::vector<uint8_t>(std::begin(channelMapping), std::begin(channelMapping) + chGroupSize),
                    0);
        }
        // otherwise store the complete channel mapping array in Us4OEM handle
        // (check the value returned by current method).

        // Other parameters
        // TGC
        // TODO replace the below calls with calls to the Us4OEM methods, i.e. remove the below code duplicate
        return std::make_unique<Us4OEMImpl>(DeviceId(DeviceType::Us4OEM, ordinal),
                                            std::move(ius4oem),
                                            channelMapping,
                                            cfg.getRxSettings(),
                                            cfg.getReprogrammingMode(),
                                            isExternalTrigger, acceptRxNops);
    }

private:

    static bool
    isConsistentPermutations(const std::vector<ChannelIdx> &vector,
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
