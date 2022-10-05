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
#include "arrus/core/api/devices/us4r/Us4OEMSettings.h"
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

    static constexpr float MIN_TX_FREQUENCY = 1e6f;
    static constexpr float MAX_TX_FREQUENCY = 60e6f;

    static constexpr float MIN_RX_TIME = 20e-6f;

    // Sampling
    static constexpr float SAMPLING_FREQUENCY = 65e6;
    static constexpr uint32_t TX_SAMPLE_DELAY_RAW_DATA = 240;
    static constexpr float RX_DELAY = 0.0;
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
    // Time to TX starting from the sample 0, when DDC is turned on. Determined experimentally.
    // Note: the below reffers to the number of IQ pairs (not the int16 values).
    static constexpr uint32_t TX_SAMPLE_DELAY_DDC_DATA[] = {
        // 1(?), 2,  3,  4,  5,  6,  7,  8,  9
          240,  92, 87, 84, 70, 60, 56, 32,  27
    };

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
               bool externalTrigger);

    ~Us4OEMImpl() override;

    bool isMaster() override;

    void startTrigger() override;

    void stopTrigger() override;

    void syncTrigger() override;

    std::tuple<Us4OEMBuffer, FrameChannelMapping::Handle>
    setTxRxSequence(const std::vector<TxRxParameters> &seq, const ops::us4r::TGCCurve &tgcSamples, uint16 rxBufferSize,
                    uint16 rxBatchSize, std::optional<float> sri, bool triggerSync = false,
                    const std::optional<::arrus::ops::us4r::DigitalDownConversion> &ddc = std::nullopt) override;

    float getSamplingFrequency() override;

    Interval<Voltage> getAcceptedVoltageRange() override {
        return Interval<Voltage>(MIN_VOLTAGE, MAX_VOLTAGE);
    }

    void start() override;

    void stop() override;

    void setTgcCurve(const RxSettings &cfg);

    Ius4OEMRawHandle getIUs4oem() override;

    void enableSequencer() override;

    std::vector<uint8_t> getChannelMapping() override;
    void setRxSettings(const RxSettings &newSettings) override;
    float getFPGATemperature() override;
    float getUCDMeasuredVoltage(uint8_t rail) override;
    void checkFirmwareVersion() override;
    uint32 getFirmwareVersion() override;
    void checkState() override;
    uint32 getTxFirmwareVersion() override;

    void setTestPattern(RxTestPattern pattern) override;

    uint16_t getAfe(uint8_t address) override;
    void setAfe(uint8_t address, uint16_t value) override;

    void setAfeDemod(const std::optional<ops::us4r::DigitalDownConversion> &ddc) {
        if(ddc.has_value()) {
            auto &value = ddc.value();
            setAfeDemod(value.getDemodulationFrequency(), value.getDecimationFactor(),
                        value.getFirCoefficients().data(), value.getFirCoefficients().size());
        }
    }

    void setAfeDemod(float demodulationFrequency, float decimationFactor, const int16_t *firCoefficients,
                     size_t nCoefficients) override {
        setAfeDemodInternal(demodulationFrequency, decimationFactor, firCoefficients, nCoefficients);
    }

    void setAfeDemod(float demodulationFrequency, float decimationFactor, const float *firCoefficients,
                     size_t nCoefficients) override {
        setAfeDemodInternal(demodulationFrequency, decimationFactor, firCoefficients, nCoefficients);
    }
    void disableAfeDemod() override {
        ius4oem->AfeDemodDisable();
    }
    float getCurrentSamplingFrequency() const override;

    float getFPGAWallclock() override;

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
    uint32_t getTxStartSampleNumberAfeDemod(float ddcDecimationFactor) const;

    // IUs4OEM AFE setters.
    void setRxSettingsPrivate(const RxSettings &newSettings, bool force = false);
    void setPgaGainAfe(uint16 value, bool force);
    void setLnaGainAfe(uint16 value, bool force);
    void setDtgcAttenuationAfe(std::optional<uint16> param, bool force);
    void setLpfCutoffAfe(uint32 value, bool force);
    void setActiveTerminationAfe(std::optional<uint16> param, bool force);
    void enableAfeDemod();
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

    template<typename T>
    void setAfeDemodInternal(float demodulationFrequency, float decimationFactor, const T *firCoefficients,
                             size_t nCoefficients) {
        //check decimation factor
        if (!(decimationFactor >= 2.0f && decimationFactor <= 63.75f)) {
            throw IllegalArgumentException("Decimation factor should be in range 2.0 - 63.75");
        }

        int decInt = static_cast<int>(decimationFactor);
        float decFract = decimationFactor - static_cast<float>(decInt);
        int nQuarters = 0;
        if (decFract == 0.0f || decFract == 0.25f || decFract == 0.5f || decFract == 0.75f) {
            nQuarters = int(decFract * 4.0f);
        } else {
            throw IllegalArgumentException("Decimation's fractional part should be equal 0.0, 0.25, 0.5 or 0.75");
        }
        int expectedNumberOfCoeffs = 0;
        //check if fir size is correct for given decimation factor
        if (nQuarters == 0) {
            expectedNumberOfCoeffs = 8 * decInt;
        }
        else if (nQuarters == 1) {
            expectedNumberOfCoeffs = 16 * decInt + 8;
        }
        else if (nQuarters == 2) {
            expectedNumberOfCoeffs = 32 * decInt + 8;
        }
        else if (nQuarters == 3) {
            expectedNumberOfCoeffs = 32 * decInt + 24;
        }
        if(static_cast<size_t>(expectedNumberOfCoeffs) != nCoefficients) {
            throw IllegalArgumentException(format("Incorrect number of DDC FIR filter coefficients, should be {}, "
                                                  "actual: {}", expectedNumberOfCoeffs, nCoefficients));
        }
        enableAfeDemod();
        //write default config
        setAfeDemodDefault();
        //set demodulation frequency
        setAfeDemodFrequency(demodulationFrequency);
        //set decimation factor
        setAfeDemodDecimationFactor(static_cast<uint8_t>(decInt), static_cast<uint8_t>(nQuarters));
        //write fir
        writeAfeFIRCoeffs(firCoefficients, static_cast<uint16_t>(nCoefficients));
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
    float currentSamplingFrequency;
    /** Global state mutex */
    mutable std::mutex stateMutex;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H
