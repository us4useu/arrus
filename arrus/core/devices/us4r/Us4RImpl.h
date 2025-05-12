#ifndef ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4RIMPL_H

#include <mutex>
#include <unordered_map>
#include <utility>
#include <thread>

#include <boost/algorithm/string.hpp>
#include <vector>

#include "BlockingQueue.h"
#include "TxTimeoutRegister.h"
#include "Us4REvent.h"
#include "Us4RSubsequence.h"
#include "arrus/common/asserts.h"
#include "arrus/common/cache.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/api/framework/DataBufferSpec.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/us4r/BlockingQueue.h"
#include "arrus/core/devices/us4r/Us4OEMDataTransferRegistrar.h"
#include "arrus/core/devices/us4r/backplane/DigitalBackplane.h"
#include "arrus/core/devices/us4r/hv/HighVoltageSupplier.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/utils.h"

namespace arrus::devices {

class Us4RImpl : public Us4R {
public:
    using Us4OEMs = std::vector<Us4OEMImplBase::Handle>;

    enum class State { START_IN_PROGRESS, STARTED, STOP_IN_PROGRESS, STOPPED };

    static float getRxDelay(const ::arrus::ops::us4r::TxRx &op);

    ~Us4RImpl() override;

    Us4RImpl(const DeviceId &id, Us4OEMs us4oems, std::vector<ProbeSettings> probeSettings,
             ProbeAdapterSettings probeAdapterSettings, std::vector<HighVoltageSupplier::Handle> hv,
             const RxSettings &rxSettings, std::vector<std::unordered_set<unsigned short>> channelsMask,
             std::optional<DigitalBackplane::Handle> backplane, std::vector<Bitstream> bitstreams,
             bool hasIOBitstreamAddressing, const us4r::IOSettings &ioSettings, bool isExternalTrigger);

    Us4RImpl(Us4RImpl const &) = delete;

    Us4RImpl(Us4RImpl const &&) = delete;

    Device::RawHandle getDevice(const std::string &path) override {
        auto [root, tail] = getPathRoot(path);
        boost::algorithm::trim(root);
        boost::algorithm::trim(tail);
        if (!tail.empty()) {
            throw IllegalArgumentException(arrus::format(
                "Us4R devices allows access only to the top-level devices (got relative path: '{}')", path));
        }
        DeviceId componentId = DeviceId::parse(root);
        return getDevice(componentId);
    }

    Device::RawHandle getDevice(const DeviceId &deviceId) {
        auto ordinal = deviceId.getOrdinal();
        switch (deviceId.getDeviceType()) {
        case DeviceType::Us4OEM: return getUs4OEM(ordinal);
        default: throw DeviceNotFoundException(deviceId);
        }
    }

    Us4OEM::RawHandle getUs4OEM(Ordinal ordinal) override {
        if (ordinal >= us4oems.size()) {
            throw DeviceNotFoundException(DeviceId(DeviceType::Us4OEM, ordinal));
        }
        return us4oems.at(ordinal).get();
    }

    bool isUs4OEMPlus() {
        return this->getMasterOEM()->getDescriptor().isUs4OEMPlus();
    }

    std::pair<Buffer::SharedHandle, std::vector<session::Metadata::SharedHandle>>
    upload(const ops::us4r::Scheme &scheme) override;

    void start() override;

    void stop() override;

    void trigger(bool sync, std::optional<long long> timeout) override;

    void sync(std::optional<long long> timeout) override;

    void setVoltage(Voltage voltage) override;
    void setVoltage(const std::vector<HVVoltage> &voltages) override;

    void disableHV() override;
    void cleanupBuffers(bool cleanupSequencerTransfers = false);

    void setTgcCurve(const std::vector<float> &tgcCurvePoints, bool applyCharacteristic) override;
    void setTgcCurve(const std::vector<float> &x, const std::vector<float> &y, bool applyCharacteristic) override;

    void setTgcCurve(const std::vector<float> &tgcCurvePoints) override;
    std::vector<float> getTgcCurvePoints(float endSample) const override;

    void setRxSettings(const RxSettings &settings) override;
    void setPgaGain(uint16 value) override;
    uint16 getPgaGain() override;
    void setLnaGain(uint16 value) override;
    uint16 getLnaGain() override;
    void setLpfCutoff(uint32 value) override;
    void setDtgcAttenuation(std::optional<uint16> value) override;
    void setActiveTermination(std::optional<uint16> value) override;
    uint8_t getNumberOfUs4OEMs() override;
    void setTestPattern(Us4OEM::RxTestPattern pattern) override;
    float getSamplingFrequency() const override;
    float getCurrentSamplingFrequency() const override;
    void checkState() const override;
    void checkVoltage(Voltage voltageMinus, Voltage voltagePlus, float tolerance, int retries, bool isUS4PSC);
    unsigned char getVoltage() override;
    float getMeasuredPVoltage() override;
    float getMeasuredMVoltage() override;
    float getMeasuredHVPVoltage(uint8_t oemId) override;
    float getMeasuredHVMVoltage(uint8_t oemId) override;
    void setStopOnOverflow(bool isStopOnOverflow) override;
    bool isStopOnOverflow() const override;
    void setLnaHpfCornerFrequency(uint32_t frequency) override;
    void disableLnaHpf() override;
    void setAdcHpfCornerFrequency(uint32_t frequency) override;
    void disableAdcHpf() override;

    uint16_t getAfe(uint8_t reg) override;
    void setAfe(uint8_t reg, uint16_t val) override;

    void registerOutputBuffer(Us4ROutputBuffer *outputBuffer, const std::vector<Us4OEMBuffer> &srcBuffers,
                              arrus::ops::us4r::Scheme::WorkMode workMode);
    void unregisterOutputBuffer(bool cleanSequencer);
    const char *getBackplaneSerialNumber() override;
    const char *getBackplaneRevision() override;
    void setParameters(const Parameters &parameters) override;
    void setIOBitstream(BitstreamId id, const std::vector<uint8_t> &levels,
                        const std::vector<uint16_t> &periods) override;
    std::vector<std::vector<uint8_t>> getOEMMappings() const;
    std::optional<Ordinal> getFrameMetadataOEM(const us4r::IOSettings &settings);

    std::vector<unsigned short> getChannelsMask(Ordinal probeNumber) override;
    int getNumberOfProbes() const override;

    Probe::RawHandle getProbe(Ordinal ordinal) override {
        return probes.at(ordinal).get();
    }

    std::pair<std::shared_ptr<Buffer>, std::shared_ptr<session::Metadata>>
    setSubsequence(SequenceId sequenceId, uint16 start, uint16 end, const std::optional<float> &sri) override;

    void setMaximumPulseLength(std::optional<float> maxLength) override;
    float getActualTxFrequency(float frequency) override;
    std::string getDescription() const override;
    float getMinimumTGCValue() const override;

    /**
     * Returns maximum available TGC value, according to the currently set parameters.
     */
    float getMaximumTGCValue() const override;

    std::pair<float, float> getTGCValueRange() const;
    void setVcat(const std::vector<float> &t, const std::vector<float> &y, bool applyCharacteristic) override;
    void setVcat(const std::vector<float> &attenuation) override;
    void setVcat(const std::vector<float> &tgcCurvePoints, bool applyCharacteristic) override;

private:
    struct VoltageLogbook {
        enum class Polarity { MINUS, PLUS };

        std::string name;
        float voltage;
        Polarity polarity;
    };


    std::vector<VoltageLogbook> logVoltages(bool isHV256);

    void stopDevice();

    std::tuple<
        std::vector<Us4OEMBuffer>,
        std::vector<FrameChannelMappingImpl::Handle>,
        float,
        std::vector<LogicalToPhysicalOp>,
        std::vector<std::vector<::arrus::devices::us4r::TxRxParametersSequence>>
    >
    uploadSequences(const std::vector<ops::us4r::TxRxSequence> &sequences, uint16_t bufferSize,
                    ops::us4r::Scheme::WorkMode workMode, const std::optional<ops::us4r::DigitalDownConversion> &ddc,
                    const std::vector<framework::NdArray> &txDelayProfiles);
    us4r::TxRxParameters createBitstreamSequenceSelectPreamble(const ops::us4r::TxRxSequence &sequence);
    std::vector<us4r::TxRxParametersSequence>
    convertToInternalSequences(
        const std::vector<ops::us4r::TxRxSequence> &sequences,
        const TxTimeoutRegister &timeoutRegister,
        const std::vector<std::vector<float>> &rxDelays
    );

    /**
     * Applies a given function on all functions.
     * If there was some exception thrown on execution of a given function,
     * an appropriate logging message will printed out, and the result exception,
     * TODO consider implementing rollback mechanism?
     */
    void applyForAllUs4OEMs(const std::function<void(Us4OEM *us4oem)> &func, const std::string &funcName);
    void disableAfeDemod();
    void setAfeDemod(float demodulationFrequency, float decimationFactor, const float *firCoefficients,
                     size_t nCoefficients);

    void registerOutputBuffer(Us4ROutputBuffer *bufferDst, const Us4OEMBuffer &bufferSrc,
                              Us4OEMImplBase::RawHandle us4oem, ::arrus::ops::us4r::Scheme::WorkMode workMode);
    size_t getUniqueUs4OEMBufferElementSize(const Us4OEMBuffer &us4oemBuffer) const;

    std::function<void()> createReleaseCallback(::arrus::ops::us4r::Scheme::WorkMode workMode, uint16 startFiring,
                                                uint16 stopFiring);
    std::function<void()> createOnReceiveOverflowCallback(::arrus::ops::us4r::Scheme::WorkMode workMode,
                                                          Us4ROutputBuffer *buffer, bool isMaster);
    std::function<void()> createOnTransferOverflowCallback(::arrus::ops::us4r::Scheme::WorkMode workMode,
                                                           Us4ROutputBuffer *buffer, bool isMaster);

    BitstreamId addIOBitstream(const std::vector<uint8_t> &levels, const std::vector<uint16_t> &periods);
    Us4OEMImplBase::RawHandle getMasterOEM() const { return this->us4oems[0].get(); }
    std::vector<float> interpolateToSystemTGC(const std::vector<float> &t, const std::vector<float> &y) const;
    void handlePulserInterrupt();
    void setVoltage(const std::vector<std::optional<HVVoltage>> &voltages);

    void prepareHostBuffer(unsigned hostBufNElements, ::arrus::ops::us4r::Scheme::WorkMode workMode, std::vector<Us4OEMBuffer> buffers,
                           bool cleanupSequencerTransfers = false);
    std::vector<arrus::session::Metadata::SharedHandle>
    createMetadata(std::vector<FrameChannelMappingImpl::Handle> fcms, float rxTimeOffset) const;

    std::mutex deviceStateMutex;
    Logger::Handle logger;
    Us4OEMs us4oems;
    std::optional<DigitalBackplane::Handle> digitalBackplane;
    std::vector<HighVoltageSupplier::Handle> hv;
    // Settings.
    State state{State::STOPPED};
    // AFE parameters.
    std::mutex afeParamsMutex;
    std::optional<RxSettings> rxSettings;
    std::vector<Probe::Handle> probes;
    std::vector<ProbeSettings> probeSettings;
    ProbeAdapterSettings probeAdapterSettings;
    std::vector<std::unordered_set<ChannelIdx>> channelsMask;
    bool stopOnOverflow{true};
    // Buffers.
    std::vector<Us4OEMBuffer> oemBuffers;
    std::shared_ptr<Us4ROutputBuffer> buffer;
    std::vector<std::shared_ptr<Us4OEMDataTransferRegistrar>> transferRegistrar;
    // Other.
    std::vector<Bitstream> bitstreams;
    bool hasIOBitstreamAdressing{false};
    std::optional<Ordinal> frameMetadataOEM{Ordinal(0)};
    bool isExternalTrigger;

    std::optional<Us4RSubsequenceFactory> subsequenceFactory;
    std::optional<Us4RSubsequence> currentSubsequenceParams;
    /** The currently uploaded scheme */
    std::optional<::arrus::ops::us4r::Scheme> currentScheme;
    std::optional<float> currentRxTimeOffset;
    std::vector<std::vector<float>> getRxDelays(const std::vector<arrus::ops::us4r::TxRxSequence> &seqs);
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
