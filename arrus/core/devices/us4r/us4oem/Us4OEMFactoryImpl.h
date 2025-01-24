#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMFACTORYIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMFACTORYIMPL_H

#include "Us4OEMDescriptorFactory.h"
#include "arrus/common/asserts.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"
#include "arrus/core/devices/us4r/external/ius4oem/PGAGainValueMap.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMFactory.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMSettingsValidator.h"

namespace arrus::devices {
class Us4OEMFactoryImpl : public Us4OEMFactory {
public:
    Us4OEMFactoryImpl() = default;

    Us4OEMImplBase::Handle getUs4OEM(Ordinal ordinal, IUs4OEMHandle &ius4oem, const Us4OEMSettings &cfg,
                                     bool isExternalTrigger, bool acceptRxNops,
                                     const std::optional<Us4RTxRxLimits> &limits) override {
        // Validate settings.
        Us4OEMDescriptor descriptor = getOEMDescriptor(ordinal, ius4oem, limits);

        Us4OEMSettingsValidator validator(ordinal, descriptor);

        validator.validate(cfg);
        validator.throwOnErrors();

        // We assume here, that the ius4oem is already initialized.
        // Configure IUs4OEM
        ChannelIdx chGroupSize = descriptor.getNRxChannels();
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
                                            descriptor,
                                            isExternalTrigger, acceptRxNops);
    }


    [[nodiscard]] static Us4OEMDescriptor getOEMDescriptor(Ordinal ordinal, const IUs4OEMHandle &ius4oem,
                                                           const std::optional<Us4RTxRxLimits> &limits) {
        auto descriptor = Us4OEMDescriptorFactory::getDescriptor(ius4oem, ordinal == 0);
        if(limits.has_value()) {
            // Update the defualt limits with the limits defined by the user.
            ops::us4r::TxRxSequenceLimitsBuilder sequenceLimitsBuilder{descriptor.getTxRxSequenceLimits()};
            ops::us4r::TxLimitsBuilder txLimitsBuilder{descriptor.getTxRxSequenceLimits().getTxRx().getTx1()};
            if(limits->getPulseLength().has_value()) {
                txLimitsBuilder.setPulseLength(limits->getPulseLength().value());
            }
            if(limits->getVoltage().has_value()) {
                txLimitsBuilder.setVoltage(limits->getVoltage().value());
            }
            Interval<float> newPri;
            if(limits->getPri().has_value()) {
                newPri = limits->getPri().value();
            }
            else {
                newPri = descriptor.getTxRxSequenceLimits().getTxRx().getPri();
            }
            auto newTxLimits = txLimitsBuilder.build();
            auto currentRxLimits = descriptor.getTxRxSequenceLimits().getTxRx().getRx();
            sequenceLimitsBuilder.setTxRxLimits(descriptor.getTxRxSequenceLimits().getTxRx().getTx0() ,newTxLimits, currentRxLimits, newPri);
            auto txRxSequenceLimits = sequenceLimitsBuilder.build();
            auto newDescriptor = Us4OEMDescriptorBuilder{descriptor}
                                     .setTxRxSequenceLimits(txRxSequenceLimits)
                                     .build();
            return newDescriptor;
        }
        else {
            return descriptor;
        }
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
