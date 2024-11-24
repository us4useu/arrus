#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H

#include "Us4OEMRxMappingRegisterBuilder.h"

#include <iostream>
#include <unordered_set>
#include <utility>

#include <ius4oem.h>
#include "IRQEvent.h"
#include "Us4OEMDescriptor.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"
#include "arrus/common/format.h"
#include "arrus/common/cache.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
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
 * This class stores reordered channels, as it is required in IUs4OEM docs.
 */
class Us4OEMImpl : public Us4OEMImplBase {
public:
    using Handle = std::unique_ptr<Us4OEMImpl>;
    using RawHandle = PtrHandle<Us4OEMImpl>;
    using FiringId = uint16;
    using RawDataType = int16;
    static constexpr framework::NdArray::DataType DataType = framework::NdArray::DataType::INT16;

    /**
     * Us4OEMImpl constructor.
     *
     * @param ius4oem
     * @param activeChannelGroups must contain exactly N_ACTIVE_CHANNEL_GROUPS elements
     * @param channelMapping a vector of N_TX_CHANNELS destination channels; must contain
     *  exactly N_TX_CHANNELS numbers
     */
    Us4OEMImpl(DeviceId id, IUs4OEMHandle ius4oem,
               std::vector<uint8_t> channelMapping, RxSettings rxSettings,
               Us4OEMSettings::ReprogrammingMode reprogrammingMode, Us4OEMDescriptor descriptor,
               bool externalTrigger, bool acceptRxNops);
    ~Us4OEMImpl() override;

    bool isMaster() override;
    void startTrigger() override;
    void stopTrigger() override;
    void syncTrigger() override;
    void setTgcCurve(const std::vector<us4r::TxRxParametersSequence> & sequences);
    Us4OEMUploadResult upload(const std::vector<us4r::TxRxParametersSequence> &sequences, uint16 rxBufferSize,
                              ops::us4r::Scheme::WorkMode workMode,
                              const std::optional<ops::us4r::DigitalDownConversion> &ddc,
                              const std::vector<arrus::framework::NdArray> &txDelays,
                              const std::vector<TxTimeout> &txTimeouts) override;

    float getSamplingFrequency() override;
    void start() override;
    void stop() override;
    Ius4OEMRawHandle getIUs4OEM() override;
    void enableSequencer(uint16_t startEntry) override;
    std::vector<uint8_t> getChannelMapping() override;
    void setRxSettings(const RxSettings &settings) override;
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
    uint32_t getOemVersion() override;
    void setTestPattern(RxTestPattern pattern) override;
    uint16_t getAfe(uint8_t address) override;
    void setAfe(uint8_t address, uint16_t value) override;
    void setAfeDemod(const std::optional<ops::us4r::DigitalDownConversion> &ddc);
    void setAfeDemod(float demodulationFrequency, float decimationFactor, const float *firCoefficients,
                     size_t nCoefficients) override;
    void disableAfeDemod() override { ius4oem->AfeDemodDisable(); }
    float getCurrentSamplingFrequency() const override;
    float getFPGAWallclock() override;
    const char *getSerialNumber() override;
    const char *getRevision() override;
    BitstreamId addIOBitstream(const std::vector<uint8_t> &levels, const std::vector<uint16_t> &lengths) override;
    void setIOBitstream(BitstreamId id, const std::vector<uint8_t> &levels,
                        const std::vector<uint16_t> &lengths) override;

    void setLnaHpfCornerFrequency(uint32_t frequency) override;
    void disableLnaHpf() override;
    void setAdcHpfCornerFrequency(uint32_t frequency) override;
    void disableAdcHpf() override;
    Interval<Voltage> getAcceptedVoltageRange() override;
    void clearCallbacks() override;
    Us4OEMDescriptor getDescriptor() const override;

    HVPSMeasurement getHVPSMeasurement() override;

    float setHVPSSyncMeasurement(uint16_t nSamples, float frequency) override;

    void setMaximumPulseLength(std::optional<float> maxLength) override;

    void sync(std::optional<long long> timeout) override;
    void setWaitForHVPSMeasurementDone() override;
    void waitForHVPSMeasurementDone(std::optional<long long> timeout) override;
    float getActualTxFrequency(float frequency) override;

    bool isOEMPlus() {
        auto version = getOemVersion();
        return version >= 2 && version <= 5;
    }

    bool isAFEJD18() override {
        return getOemVersion() == 1 || getOemVersion() == 2;
    }

    bool isAFEJD48() override {
        return getOemVersion() == 3;
    }

    std::pair<float, float> getTGCValueRange() const override;

private:
    using Us4OEMAperture = std::bitset<Us4OEMDescriptor::N_ADDR_CHANNELS>;
    using Us4OEMChannelsGroupsMask = std::bitset<Us4OEMDescriptor::N_ACTIVE_CHANNEL_GROUPS>;

    float getTxRxTime(float rxTime) const;
    float getRxTime(const ::arrus::devices::us4r::TxRxParameters &op, float samplingFrequency);
    /**
     * Returns the sample number that corresponds to the time of Tx.
     */
    std::pair<uint32_t, float> getTxStartSampleNumberAfeDemod(float ddcDecimationFactor);

    // IUs4OEM AFE setters.
    void enableAfeDemod();
    void setAfeDemodConfig(uint8_t decInt, uint8_t decQuarters, const float *firCoeffs, uint16_t firLength, float freq);
    void setAfeDemodDefault();
    void setAfeDemodDecimationFactor(uint8_t integer);
    void setAfeDemodDecimationFactor(uint8_t integer, uint8_t quarters);
    void setAfeDemodFrequency(float frequency);
    void setAfeDemodFrequency(float StartFrequency, float stopFrequency);
    float getAfeDemodStartFrequency();
    float getAfeDemodStopFrequency();
    void setAfeDemodFsweepROI(uint16_t startSample, uint16_t stopSample);
    void writeAfeFIRCoeffs(const int16_t *coeffs, uint16_t length);
    void writeAfeFIRCoeffs(const float *coeffs, uint16_t length);
    void resetAfe();
    void setIOBitstreamForOffset(uint16 bitstreamOffset, const std::vector<uint8_t> &levels,
                                 const std::vector<uint16_t> &periods);
    void setCurrentSamplingFrequency(float fs) { this->currentSamplingFrequency = fs; }
    void setTxDelays(const std::vector<bool> &txAperture, const std::vector<float> &delays, uint16 firingId, size_t delaysId,
                     const std::unordered_set<ChannelIdx> &maskedChannelsTx);
    void setTgcCurve(const ops::us4r::TGCCurve &tgc);
    Us4OEMChannelsGroupsMask getActiveChannelGroups(const Us4OEMAperture &txAperture, const Us4OEMAperture &rxAperture);
    void uploadFirings(const us4r::TxParametersSequenceColl &sequences,
                       const std::optional<ops::us4r::DigitalDownConversion> &ddc,
                       const std::vector<arrus::framework::NdArray> &txDelays,
                       const Us4OEMRxMappingRegister &rxMappingRegister);
    std::pair<size_t, float> scheduleReceiveDDC(size_t outputAddress,
                                                uint32 startSample, uint32 endSample, uint16 entryId,
                                                const us4r::TxRxParameters &op, uint16 rxMapId,
                                                const std::optional<ops::us4r::DigitalDownConversion> &ddc);
    size_t scheduleReceiveRF(size_t outputAddress, uint32 startSample, uint32 endSample, uint16 entryId,
                             const us4r::TxRxParameters &op, uint16 rxMapId);
    std::pair<Us4OEMBuffer, float> uploadAcquisition(const us4r::TxParametersSequenceColl &sequences, uint16 rxBufferSize,
                                                     const std::optional<ops::us4r::DigitalDownConversion> &ddc,
                                                     const Us4OEMRxMappingRegister &rxMappingRegister);
    void uploadTriggersIOBS(const us4r::TxParametersSequenceColl &sequences, uint16 rxBufferSize,
                            ops::us4r::Scheme::WorkMode workMode);
    void waitForIrq(unsigned int irq, std::optional<long long> timeout);

    void validate(const us4r::TxParametersSequenceColl &sequences, uint16 rxBufferSize);
    size_t getNumberOfFirings(const us4r::TxParametersSequenceColl &vector);
    size_t getNumberOfTriggers(const us4r::TxParametersSequenceColl &sequences, uint16 rxBufferSize);
    Us4OEMRxMappingRegister setRxMappings(const us4r::TxParametersSequenceColl &sequences);
    void setWaitForEventDone();
    std::bitset<Us4OEMDescriptor::N_ADDR_CHANNELS> filterAperture(
        std::bitset<Us4OEMDescriptor::N_ADDR_CHANNELS> aperture,
        const std::unordered_set<ChannelIdx> &channelsMask);

    Logger::Handle logger;
    IUs4OEMHandle ius4oem;
    Us4OEMDescriptor descriptor;
    // Tx channel mapping (and Rx implicitly): logical channel -> physical channel
    std::vector<uint8_t> channelMapping;
    Us4OEMSettings::ReprogrammingMode reprogrammingMode;
    /** Current RX settings */
    // TODO(ARRUS-179) consider removing the below property
    RxSettings rxSettings;
    bool externalTrigger{false};
    /** Current sampling frequency of the data produced by us4OEM. */
    float currentSamplingFrequency{0};
    /** Global state mutex */
    mutable std::mutex stateMutex;
    arrus::Cached<std::string> serialNumber;
    arrus::Cached<std::string> revision;
    bool acceptRxNops{false};
    /** maps from the bitstream id (0, 1,...), to bitstream FPGA offset. */
    std::vector<uint16> bitstreamOffsets;
    /** The size of each bitstream defined (the number of registers). */
    std::vector<uint16> bitstreamSizes;
    bool isDecimationFactorAdjustmentLogged{false};
    std::vector<IRQEvent> irqEvents = std::vector<IRQEvent>(Us4OEMDescriptor::MAX_IRQ_NR+1);
    /** Max TX pulse length [s]; nullopt means to use up to 32 periods (OEM legacy constraint) */
    std::optional<float> maxPulseLength = std::nullopt;
    void setTxTimeouts(const std::vector<TxTimeout> &txTimeouts);
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEMIMPL_H
