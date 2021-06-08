#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H

#include <utility>
#include <iostream>
#include <unordered_set>

#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
#include "arrus/common/format.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/framework/NdArray.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/devices/UltrasoundDevice.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImplBase.h"
#include "arrus/core/devices/us4r/DataTransfer.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMBuffer.h"


namespace arrus::devices {

/**
 * Us4OEM wrapper implementation.
 *
 * This class stores reordered channels, as it is required in IUs4OEM docs.
 */
class Us4OEMImpl : public Us4OEMImplBase {
public:
    using Handle = std::unique_ptr<Us4OEMImpl>;
    using RawHandle = PtrHandle<Us4OEMImpl>;

    using FiringIdx = uint16;
    using OutputDType = int16;
    static constexpr framework::NdArray::DataType NdArrayDataType = framework::NdArray::DataType::INT16;

    // voltage, +/- [V] amplitude, (ref: technote)
    static constexpr Voltage MIN_VOLTAGE = 0;
    static constexpr Voltage MAX_VOLTAGE = 90; // 180 vpp

    // TGC constants.
    static constexpr float TGC_ATTENUATION_RANGE = 40.0f;
    static constexpr float TGC_SAMPLING_FREQUENCY = 1e6;
    static constexpr size_t TGC_N_SAMPLES = 1022;

    // Number of tx/rx channels.
    static constexpr ChannelIdx N_TX_CHANNELS = 128;
    static constexpr ChannelIdx N_RX_CHANNELS = 32;
    static constexpr ChannelIdx N_ADDR_CHANNELS = N_TX_CHANNELS;
    static constexpr ChannelIdx ACTIVE_CHANNEL_GROUP_SIZE = 8;
    static constexpr ChannelIdx N_ACTIVE_CHANNEL_GROUPS =
        N_TX_CHANNELS / ACTIVE_CHANNEL_GROUP_SIZE;

    static constexpr float MIN_TX_DELAY = 0.0f;
    static constexpr float MAX_TX_DELAY = 16.96e-6f;

    static constexpr float MIN_TX_FREQUENCY = 1e6f;
    static constexpr float MAX_TX_FREQUENCY = 60e6f;

    // Sampling
    static constexpr float SAMPLING_FREQUENCY = 65e6;
    static constexpr uint32 SAMPLE_DELAY = 240;
    static constexpr float RX_DELAY = 0.0;
    static constexpr float RX_TIME_EPSILON = 5e-6f;
    static constexpr uint32 MIN_NSAMPLES = 64;
    static constexpr uint32 MAX_NSAMPLES = 16384;
    // Data
    static constexpr size_t DDR_SIZE = 1ull << 32u;
    // Other
    static constexpr float MIN_PRI = 100e-6f; // [s]
    static constexpr float MIN_RX_TIME = 60e-6f; // [s]
    static constexpr float SEQUENCER_REPROGRAMMING_TIME = 35e-6f; // [s]
    static constexpr float MAX_PRI = 1.0f;

    /**
     * Us4OEMImpl constructor.
     *
     * @param ius4oem
     * @param activeChannelGroups must contain exactly N_ACTIVE_CHANNEL_GROUPS elements
     * @param channelMapping a vector of N_TX_CHANNELS destination channels; must contain
     *  exactly N_TX_CHANNELS numbers
     */
    Us4OEMImpl(DeviceId id, IUs4OEMHandle ius4oem,
               const BitMask &activeChannelGroups,
               std::vector<uint8_t> channelMapping,
               uint16 pgaGain, uint16 lnaGain,
               std::unordered_set<uint8_t> channelsMask);

    ~Us4OEMImpl() override;

    bool isMaster() override;

    void startTrigger() override;

    void stopTrigger() override;

    void syncTrigger() override;

    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle>
    setTxRxSequence(const std::vector<TxRxParameters> &seq,
                    const ops::us4r::TGCCurve &tgcSamples, uint16 rxBufferSize,
                    uint16 rxBatchSize, std::optional<float> sri,
                    bool triggerSync = false) override;

    double getSamplingFrequency() override;

    Interval<Voltage> getAcceptedVoltageRange() override {
        return Interval<Voltage>(MIN_VOLTAGE, MAX_VOLTAGE);
    }

    void start() override;

    void stop() override;

    void setTgcCurve(const ops::us4r::TGCCurve &tgc) override;

    Ius4OEMRawHandle getIUs4oem() override;

    void enableSequencer() override;

    std::vector<uint8_t> getChannelMapping() override;

 private:
    using Us4OEMBitMask = std::bitset<Us4OEMImpl::N_ADDR_CHANNELS>;

    Logger::Handle logger;
    IUs4OEMHandle ius4oem;
    std::bitset<N_ACTIVE_CHANNEL_GROUPS> activeChannelGroups;
    // Tx channel mapping (and Rx implicitly): logical channel -> physical channel
    std::vector<uint8_t> channelMapping;
    std::unordered_set<uint8_t> channelsMask;
    uint16 pgaGain, lnaGain;

    std::tuple<
        std::unordered_map<uint16, uint16>,
        std::vector<Us4OEMImpl::Us4OEMBitMask>,
        FrameChannelMapping::Handle>
    setRxMappings(const std::vector<TxRxParameters> &seq);

    static float getRxTime(size_t nSamples, uint32 decimationFactor);

    void setTGC(const ops::us4r::TGCCurve &tgc);

    std::bitset<N_ADDR_CHANNELS>
    filterAperture(std::bitset<N_ADDR_CHANNELS> aperture);

    void validateAperture(const std::bitset<N_ADDR_CHANNELS> &aperture);
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H
