#ifndef ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4RIMPL_H

#include <mutex>
#include <unordered_map>
#include <utility>

#include <boost/algorithm/string.hpp>

#include "arrus/common/asserts.h"
#include "arrus/common/cache.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/api/framework/DataBufferSpec.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/probe/ProbeImplBase.h"
#include "arrus/core/devices/us4r/RxSettings.h"
#include "arrus/core/devices/us4r/Us4OEMDataTransferRegistrar.h"
#include "arrus/core/devices/us4r/Us4RBuffer.h"
#include "arrus/core/devices/us4r/backplane/DigitalBackplane.h"
#include "arrus/core/devices/us4r/hv/HighVoltageSupplier.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterImplBase.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/utils.h"

namespace arrus::devices {

class Us4RImpl : public Us4R {
public:
    using Us4OEMs = std::vector<Us4OEMImplBase::Handle>;

    enum class State { START_IN_PROGRESS, STARTED, STOP_IN_PROGRESS, STOPPED };

    ~Us4RImpl() override;

    Us4RImpl(const DeviceId &id, Us4OEMs us4oems, std::vector<HighVoltageSupplier::Handle> hv,
             std::vector<unsigned short> channelsMask,
             std::optional<DigitalBackplane::Handle> backplane
             );

    Us4RImpl(const DeviceId &id, Us4OEMs us4oems, ProbeAdapterImplBase::Handle &probeAdapter,
             ProbeImplBase::Handle &probe, std::vector<HighVoltageSupplier::Handle> hv, const RxSettings &rxSettings,
             std::vector<unsigned short> channelsMask,
             std::optional<DigitalBackplane::Handle> backplane
             );

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
        case DeviceType::ProbeAdapter: return getProbeAdapter(ordinal);
        case DeviceType::Probe: return getProbe(ordinal);
        default: throw DeviceNotFoundException(deviceId);
        }
    }

    Us4OEM::RawHandle getUs4OEM(Ordinal ordinal) override {
        if (ordinal >= us4oems.size()) {
            throw DeviceNotFoundException(DeviceId(DeviceType::Us4OEM, ordinal));
        }
        return us4oems.at(ordinal).get();
    }

    ProbeAdapter::RawHandle getProbeAdapter(Ordinal ordinal) override {
        if (ordinal > 0 || !probeAdapter.has_value()) {
            throw DeviceNotFoundException(DeviceId(DeviceType::ProbeAdapter, ordinal));
        }
        return probeAdapter.value().get();
    }

    Probe::RawHandle getProbe(Ordinal ordinal) override {
        if (ordinal > 0 || !probe.has_value()) {
            throw DeviceNotFoundException(DeviceId(DeviceType::Probe, ordinal));
        }
        return probe.value().get();
    }

    Us4OEMImplBase::RawHandle getMasterOEM() const { return this->us4oems[0].get(); }

    bool isUs4OEMPlus() {
        return getMasterOEM()->getOemVersion() == 2;
    }

    std::pair<std::shared_ptr<arrus::framework::Buffer>, std::shared_ptr<arrus::session::Metadata>>
    upload(const ::arrus::ops::us4r::Scheme &scheme) override;
    void prepareHostBuffer(unsigned nElements, ops::us4r::Scheme::WorkMode workMode,
                           std::unique_ptr<Us4RBuffer> &rxBuffer, bool cleanupSequencer = false);

    void start() override;

    void stop() override;

    void trigger(bool sync, std::optional<long long> timeout) override;

    void sync(std::optional<long long> timeout) override;

    void setVoltage(Voltage voltage) override;

    void disableHV() override;

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
    std::vector<unsigned short> getChannelsMask() override;
    std::vector<std::pair <std::string,float>> logVoltages(bool isUS4PSC);
    void checkVoltage(Voltage voltage, float tolerance, int retries, bool isUS4PSC);
    unsigned char getVoltage() override;
    float getMeasuredPVoltage() override;
    float getMeasuredMVoltage() override;
    float getMeasuredHVPVoltage(uint8_t oemId) override;
    float getMeasuredHVMVoltage(uint8_t oemId) override;
    void setStopOnOverflow(bool isStopOnOverflow) override;
    bool isStopOnOverflow() const override;
    void setHpfCornerFrequency(uint32_t frequency) override;
    void disableHpf() override;

    uint16_t getAfe(uint8_t reg) override;
    void setAfe(uint8_t reg, uint16_t val) override;

    void registerOutputBuffer(Us4ROutputBuffer *buffer, const Us4RBuffer::Handle &us4rBuffer,
                              ::arrus::ops::us4r::Scheme::WorkMode workMode);
    void unregisterOutputBuffer(bool cleanSequencer);
    const char *getBackplaneSerialNumber() override;
    const char *getBackplaneRevision() override;
    void setParameters(const Parameters &parameters) override;

    std::pair<std::shared_ptr<Buffer>, std::shared_ptr<session::Metadata>>
    setSubsequence(uint16 start, uint16 end, const std::optional<float> &sri) override;

    void setMaximumPulseLength(std::optional<float> maxLength) override;

private:
    UltrasoundDevice *getDefaultComponent();

    void stopDevice();

    std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
    uploadSequence(const ops::us4r::TxRxSequence &seq, uint16_t bufferSize, uint16_t batchSize,
                   arrus::ops::us4r::Scheme::WorkMode workMode,
                   const std::optional<ops::us4r::DigitalDownConversion> &ddc,
                   const std::vector<framework::NdArray> &txDelayProfiles);

    /**
     * Applies a given function on all functions.
     * If there was some exception thrown on execution of a given function,
     * an appropriate logging message will printed out, and the result exception,
     * TODO consider implementing rollback mechanism?
     */
    void applyForAllUs4OEMs(const std::function<void(Us4OEM* us4oem)>& func, const std::string &funcName);
    void disableAfeDemod();
    void setAfeDemod(float demodulationFrequency, float decimationFactor, const float *firCoefficients,
                     size_t nCoefficients);

    ProbeImplBase::RawHandle getProbeImpl() { return probe.value().get(); }

    void registerOutputBuffer(Us4ROutputBuffer *bufferDst, const Us4OEMBuffer &bufferSrc,
                              Us4OEMImplBase::RawHandle us4oem, ::arrus::ops::us4r::Scheme::WorkMode workMode);
    size_t getUniqueUs4OEMBufferElementSize(const Us4OEMBuffer &us4oemBuffer) const;

    std::function<void()> createReleaseCallback(
        ::arrus::ops::us4r::Scheme::WorkMode workMode, uint16 startFiring, uint16 stopFiring);
    std::function<void()> createOnReceiveOverflowCallback(
        ::arrus::ops::us4r::Scheme::WorkMode workMode, Us4ROutputBuffer *buffer, bool isMaster);
    std::function<void()> createOnTransferOverflowCallback(
        ::arrus::ops::us4r::Scheme::WorkMode workMode, Us4ROutputBuffer *buffer, bool isMaster);

    Us4OEMImplBase::RawHandle getMasterUs4oem() const {return this->us4oems[0].get();}

    std::mutex deviceStateMutex;
    std::mutex afeParamsMutex;
    Logger::Handle logger;
    Us4OEMs us4oems;
    std::optional<ProbeAdapterImplBase::Handle> probeAdapter;
    std::optional<ProbeImplBase::Handle> probe;
    std::optional<DigitalBackplane::Handle> digitalBackplane;
    std::vector<HighVoltageSupplier::Handle> hv;
    std::unique_ptr<Us4RBuffer> us4rBuffer;
    std::shared_ptr<Us4ROutputBuffer> buffer;
    State state{State::STOPPED};
    // AFE parameters.
    std::optional<RxSettings> rxSettings;
    std::vector<unsigned short> channelsMask;
    bool stopOnOverflow{true};
    std::vector<std::shared_ptr<Us4OEMDataTransferRegistrar>> transferRegistrar;
    /** Currently uploaded scheme. */
    std::optional<ops::us4r::Scheme> currentScheme;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
