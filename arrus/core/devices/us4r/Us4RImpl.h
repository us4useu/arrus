#ifndef ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4RIMPL_H

#include <mutex>
#include <unordered_map>
#include <utility>

#include <boost/algorithm/string.hpp>

#include "arrus/common/asserts.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/framework/Buffer.h"
#include "arrus/core/api/framework/DataBufferSpec.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/probe/ProbeImplBase.h"
#include "arrus/core/devices/us4r/RxSettings.h"
#include "arrus/core/devices/us4r/Us4RBuffer.h"
#include "arrus/core/devices/us4r/hv/HighVoltageSupplier.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterImplBase.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/utils.h"

namespace arrus::devices {

class Us4RImpl : public Us4R {
public:
    using Us4OEMs = std::vector<Us4OEMImplBase::Handle>;

    enum class State { STARTED, STOPPED };

    ~Us4RImpl() override;

    Us4RImpl(const DeviceId &id, Us4OEMs us4oems, std::optional<HighVoltageSupplier::Handle> hv,
             std::vector<unsigned short> channelsMask);

    Us4RImpl(const DeviceId &id, Us4OEMs us4oems, ProbeAdapterImplBase::Handle &probeAdapter,
             ProbeImplBase::Handle &probe, std::optional<HighVoltageSupplier::Handle> hv, const RxSettings &rxSettings,
             std::vector<unsigned short> channelsMask);

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

    std::pair<std::shared_ptr<arrus::framework::Buffer>,std::shared_ptr<arrus::devices::FrameChannelMapping>>
    upload(const ::arrus::ops::us4r::Scheme &scheme) override;

    void start() override;

    void stop() override;

    void trigger();

    void setVoltage(Voltage voltage) override;

    void disableHV() override;

    void setTgcCurve(const std::vector<float> &tgcCurvePoints, bool applyCharacteristic) override;

    void setTgcCurve(const std::vector<float> &tgcCurvePoints) override;

    void setRxSettings(const RxSettings &settings) override;
    void setPgaGain(uint16 value) override;
    void setLnaGain(uint16 value) override;
    void setLpfCutoff(uint32 value) override;
    void setDtgcAttenuation(std::optional<uint16> value) override;
    void setActiveTermination(std::optional<uint16> value) override;
    uint8_t getNumberOfUs4OEMs() override;
    void setTestPattern(Us4OEM::RxTestPattern pattern) override;
    float getSamplingFrequency() const override;
    void checkState() const override;
    std::vector<unsigned short> getChannelsMask() override;
    unsigned char getVoltage() override;
    float getMeasuredPVoltage() override;
    float getMeasuredMVoltage() override;
    void setStopOnOverflow(bool isStopOnOverflow) override;
    bool isStopOnOverflow() const override;


private:
    UltrasoundDevice *getDefaultComponent();

    void stopDevice();

    std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
    uploadSequence(const ops::us4r::TxRxSequence &seq, uint16_t bufferSize, uint16_t batchSize, bool triggerSync,
                   const std::optional<ops::us4r::DigitalDownConversion> &ddc);

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

    std::mutex deviceStateMutex;
    std::mutex afeParamsMutex;
    Logger::Handle logger;
    Us4OEMs us4oems;
    std::optional<ProbeAdapterImplBase::Handle> probeAdapter;
    std::optional<ProbeImplBase::Handle> probe;
    std::optional<HighVoltageSupplier::Handle> hv;
    std::unique_ptr<Us4RBuffer> us4rBuffer;
    std::shared_ptr<Us4ROutputBuffer> buffer;
    State state{State::STOPPED};
    // AFE parameters.
    std::optional<RxSettings> rxSettings;
    std::vector<unsigned short> channelsMask;
    bool stopOnOverflow{true};
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
