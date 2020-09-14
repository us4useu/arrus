#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H

#include <utility>
#include <iostream>

#include "arrus/common/format.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"
#include "arrus/core/api/ops/us4r/tgc.h"


namespace arrus::devices {

/**
 * Us4OEM wrapper implementation.
 *
 * This class stores reordered channels, as it is required in IUs4OEM docs.
 */
class Us4OEMImpl : public Us4OEM {
public:
    // TGC constants.
    static constexpr float TGC_ATTENUATION_RANGE = 40.0f;
    static constexpr float TGC_SAMPLING_FREQUENCY = 1e6;
    static constexpr size_t TGC_N_SAMPLES = 1022;

    // Number of tx/rx channels.
    static constexpr ChannelIdx N_TX_CHANNELS = 128;
    static constexpr ChannelIdx N_RX_CHANNELS = 32;
    static constexpr ChannelIdx N_RX_ADDRESABLE_CHANNELS = N_TX_CHANNELS;
    static constexpr ChannelIdx ACTIVE_CHANNEL_GROUP_SIZE = 8;
    static constexpr ChannelIdx N_ACTIVE_CHANNEL_GROUPS =
        N_TX_CHANNELS / ACTIVE_CHANNEL_GROUP_SIZE;

    // Sampling
    static constexpr float SAMPLING_FREQUENCY = 65e6;
    static constexpr uint32 TRIGGER_DELAY = 240;
    static constexpr float RX_DELAY = 0.0;
    static constexpr float RX_TIME_EPSILON = 10e-6;

    using OutputDtype = int16;


    Us4OEMImpl(DeviceId id, IUs4OEMHandle ius4oem,
               const BitMask &activeChannelGroups,
               std::vector<uint8_t> channelMapping);

    ~Us4OEMImpl() override;

    void startTrigger() override;

    void stopTrigger() override;

    void setTxRxSequence(const std::vector<TxRxParameters> &seq,
                         const ::arrus::ops::us4r::TGCCurve &tgcSamples);

    double getSamplingFrequency() override;


private:
    using Us4rBitMask = std::bitset<Us4OEMImpl::N_RX_ADDRESABLE_CHANNELS>;

    Logger::Handle logger;
    IUs4OEMHandle ius4oem;
    std::bitset<N_ACTIVE_CHANNEL_GROUPS> activeChannelGroups;
    std::vector<uint8_t> channelMapping;

    std::pair<
        std::unordered_map<uint16, uint16>,
        std::vector<Us4OEMImpl::Us4rBitMask>>
    setRxMappings(const std::vector<TxRxParameters> &seq);

    static float getRxTime(size_t nSamples);

    void setTGC(const ops::us4r::TGCCurve &tgc, uint16 firing);
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H
