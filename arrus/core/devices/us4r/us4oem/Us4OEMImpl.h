#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H

#include <utility>
#include <iostream>
#include <unordered_set>
#include <ius4oem.h>

#include "IRQEvent.h"
#include "arrus/common/cache.h"
#include "arrus/common/format.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
#include "arrus/core/api/framework/NdArray.h"
#include "arrus/core/api/ops/us4r/tgc.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/TxRxParameters.h"
#include "arrus/core/devices/UltrasoundDevice.h"
#include "arrus/core/devices/us4r/DataTransfer.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMBuffer.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImplBase.h"

namespace arrus::devices {



/**
 * Us4OEM wrapper implementation.
 *
 * Note: the current implementation assumes the first revision of us4OEM.
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
    static constexpr float TGC_ATTENUATION_RANGE = RxSettings::TGC_ATTENUATION_RANGE;
    static constexpr float TGC_SAMPLING_FREQUENCY = 1e6;
    static constexpr size_t TGC_N_SAMPLES = 1022;

    // Number of tx/rx channels.
    static constexpr ChannelIdx N_TX_CHANNELS = 128;
    static constexpr ChannelIdx N_RX_CHANNELS = 32;
    static constexpr ChannelIdx N_ADDR_CHANNELS = N_TX_CHANNELS;
    static constexpr ChannelIdx ACTIVE_CHANNEL_GROUP_SIZE = 8;
    static constexpr ChannelIdx N_ACTIVE_CHANNEL_GROUPS = N_TX_CHANNELS / ACTIVE_CHANNEL_GROUP_SIZE;

    static constexpr float MIN_TX_DELAY = 0.0f;
    static constexpr float MAX_TX_DELAY = 16.96e-6f;

    static constexpr int DEFAULT_TX_FREQUENCY_RANGE = 1;
    static constexpr float MIN_TX_FREQUENCY = 1e6f;
    static constexpr float MAX_TX_FREQUENCY = 60e6f;

    static constexpr float MIN_RX_TIME = 20e-6f;

    // Sampling
    static constexpr float SAMPLING_FREQUENCY = 65e6;
    static constexpr uint32 MIN_NSAMPLES = 64;
    static constexpr uint32 MAX_NSAMPLES = 16384;
    // Data
    static constexpr size_t DDR_SIZE = 1ull << 32u;
    static constexpr float SEQUENCER_REPROGRAMMING_TIME = 35e-6f; // [s]
    static constexpr float MIN_PRI = SEQUENCER_REPROGRAMMING_TIME;
    static constexpr float MAX_PRI = 1.0f; // [s]
    static constexpr float RX_TIME_EPSILON = 5e-6f; // [s]
    // 2^14 descriptors * 2^12 (4096, minimum page size) bytes
    static constexpr size_t MAX_TRANSFER_SIZE = 1ull << (14+12); // bytes
    static constexpr unsigned MAX_IRQ_NR = IUs4OEM::MAX_IRQ_NR;

    /**
     * Us4OEMImpl constructor.
     *
     * @param ius4oem
     * @param activeChannelGroups must contain exactly N_ACTIVE_CHANNEL_GROUPS elements
     * @param channelMapping a vector of N_TX_CHANNELS destination channels; must contain
     *  exactly N_TX_CHANNELS numbers
     */
    Us4OEMImpl(DeviceId id, IUs4OEMHandle ius4oem, const BitMask &activeChannelGroups,
               std::vector<uint8_t> channelMapping, RxSettings rxSettings,
               std::unordered_set<uint8_t> channelsMask, Us4OEMSettings::ReprogrammingMode reprogrammingMode,
               bool externalTrigger, bool acceptRxNops);

    ~Us4OEMImpl() override;

    bool isMaster() override;

    void startTrigger() override;

    void stopTrigger() override;

    void syncTrigger() override;

    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle>
    setTxRxSequence(const std::vector<TxRxParameters> &seq, const ops::us4r::TGCCurve &tgcSamples, uint16 rxBufferSize,
                    uint16 rxBatchSize, std::optional<float> sri, arrus::ops::us4r::Scheme::WorkMode workMode,
                    const std::optional<::arrus::ops::us4r::DigitalDownConversion> &ddc = std::nullopt,
                    const std::vector<arrus::framework::NdArray> &txDelays = std::vector<arrus::framework::NdArray>()
                    ) override;

    float getSamplingFrequency() override;

    Interval<Voltage> getAcceptedVoltageRange() override {
        return Interval<Voltage>(MIN_VOLTAGE, MAX_VOLTAGE);
    }

    void start() override;

    void stop() override;

    void setTgcCurve(const RxSettings &cfg);

    Ius4OEMRawHandle getIUs4oem() override;

    void enableSequencer(uint16_t startEntry) override;

    std::vector<uint8_t> getChannelMapping() override;
    void setRxSettings(const RxSettings &newSettings) override;
    float getFPGATemperature() override;
    float getUCDTemperature() override;
    float getUCDExternalTemperature() override;
    float getUCDMeasuredVoltage(uint8_t rail) override;
    float getMeasuredHVPVoltage() override;
    float getMeasuredHVMVoltage() override;
    void checkFirmwareVersion() override;
    uint32 getFirmwareVersion() override;
    void checkState() override;
    uint32 getTxFirmwareVersion() override;
    uint32_t getTxOffset() override;
    uint32_t getOemVersion() override;

    void setTestPattern(RxTestPattern pattern) override;

    uint16_t getAfe(uint8_t address) override;
    void setAfe(uint8_t address, uint16_t value) override;

    void setAfeDemod(const std::optional<ops::us4r::DigitalDownConversion> &ddc);

    void setAfeDemod(float demodulationFrequency, float decimationFactor, const float *firCoefficients,
                     size_t nCoefficients) override;

    void disableAfeDemod() override {
        ius4oem->AfeDemodDisable();
    }
    float getCurrentSamplingFrequency() const override;

    float getFPGAWallclock() override;

    const char *getSerialNumber() override;

    const char *getRevision() override;

    void setSubsequence(uint16 start, uint16 end, bool syncMode, const std::optional<float> &sri) override;

    void clearCallbacks() override;


    HVPSMeasurement getHVPSMeasurement() override;

    float setHVPSSyncMeasurement(uint16_t nSamples, float frequency) override;

    void setMaximumPulseLength(std::optional<float> maxLength) override;

    void sync(std::optional<long long> timeout) override;
    void setWaitForHVPSMeasurementDone() override;
    void waitForHVPSMeasurementDone(std::optional<long long> timeout) override;

 private:
    using Us4OEMBitMask = std::bitset<Us4OEMImpl::N_ADDR_CHANNELS>;

    std::tuple<std::unordered_map<uint16, uint16>, std::vector<Us4OEMImpl::Us4OEMBitMask>, FrameChannelMapping::Handle>
    setRxMappings(const std::vector<TxRxParameters> &seq);

    static float getRxTime(size_t nSamples, float samplingFrequency);

    std::bitset<N_ADDR_CHANNELS> filterAperture(std::bitset<N_ADDR_CHANNELS> aperture);

    void validateAperture(const std::bitset<N_ADDR_CHANNELS> &aperture);

    float getTxRxTime(float rxTime) const;

    /**
     * Returns the sample number that corresponds to the time of Tx.
     */
    uint32_t getTxStartSampleNumberAfeDemod(float ddcDecimationFactor);

    // IUs4OEM AFE setters.
    void setRxSettingsPrivate(const RxSettings &newSettings, bool force = false);
    void setPgaGainAfe(uint16 value, bool force);
    void setLnaGainAfe(uint16 value, bool force);
    void setDtgcAttenuationAfe(std::optional<uint16> param, bool force);
    void setLpfCutoffAfe(uint32 value, bool force);
    void setActiveTerminationAfe(std::optional<uint16> param, bool force);
    void enableAfeDemod();
    void setAfeDemodConfig(uint8_t decInt, uint8_t decQuarters, const float* firCoeffs, uint16_t firLength, float freq);
    void setAfeDemodDefault();
    void setAfeDemodDecimationFactor(uint8_t integer);
    void setAfeDemodDecimationFactor(uint8_t integer, uint8_t quarters);
    void setAfeDemodFrequency(float frequency);
    void setAfeDemodFrequency(float StartFrequency, float stopFrequency);
    float getAfeDemodStartFrequency(void);
    float getAfeDemodStopFrequency(void);
    void setAfeDemodFsweepROI(uint16_t startSample, uint16_t stopSample);
    void writeAfeFIRCoeffs(const int16_t* coeffs, uint16_t length);
    void writeAfeFIRCoeffs(const float* coeffs, uint16_t length);
    void resetAfe();
    void setHpfCornerFrequency(uint32_t frequency);
    void disableHpf();
    void waitForIrq(unsigned int irq, std::optional<long long> timeout);

    uint32_t getTimeToNextTrigger(float pri) {
        return static_cast<uint32_t>(std::round(pri * 1e6));
    }

    std::optional<float> getLastPriExtend(const std::vector<TxRxParameters>::const_iterator &start,
                                          const std::vector<TxRxParameters>::const_iterator &end,
                                          std::optional<float> sri) {
        float totalPri = std::accumulate(start, end, 0.0f, [](const auto &a, const auto &b) {return a + b.getPri();});
        std::optional<float> lastPriExtend = std::nullopt;
        // Sequence repetition interval.
        if (sri.has_value()) {
            if (totalPri < sri.value()) {
                lastPriExtend = sri.value() - totalPri;
            } else {
                throw IllegalArgumentException(format("Sequence repetition interval {} cannot be set, "
                                                      "sequence total pri is equal {}",
                                                      sri.value(), totalPri));
            }
        }
        return lastPriExtend;
    }

    Logger::Handle logger;
    IUs4OEMHandle ius4oem;
    std::bitset<N_ACTIVE_CHANNEL_GROUPS> activeChannelGroups;
    // Tx channel mapping (and Rx implicitly): logical channel -> physical channel
    std::vector<uint8_t> channelMapping;
    std::unordered_set<uint8_t> channelsMask;
    Us4OEMSettings::ReprogrammingMode reprogrammingMode;
    /** Current RX settings */
    RxSettings rxSettings;
    bool externalTrigger{false};
    /** Current sampling frequency of the data produced by us4OEM. */
    float currentSamplingFrequency{SAMPLING_FREQUENCY};
    /** Global state mutex */
    mutable std::mutex stateMutex;
    arrus::Cached<std::string> serialNumber;
    arrus::Cached<std::string> revision;
    bool acceptRxNops{false};
    bool isDecimationFactorAdjustmentLogged{false};
    /** Currently uploaded sequence; empty when no sequence haven't been uploaded. */
    std::vector<TxRxParameters> currentSequence;
    /** Conditional variable that is set when an IRQ with given number is detected. */

    std::vector<IRQEvent> irqEvents = std::vector<IRQEvent>(IUs4OEM::MAX_IRQ_NR+1);
    /** Max TX pulse length [s]; nullopt means to use up to 32 periods (OEM legacy constraint) */
    std::optional<float> maxPulseLength = std::nullopt;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H
