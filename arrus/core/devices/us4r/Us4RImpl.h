#ifndef ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4RIMPL_H

#include <unordered_map>
#include <utility>

#include <boost/algorithm/string.hpp>
#include <arrus/core/api/devices/us4r/HostBuffer.h>

#include "arrus/common/asserts.h"
#include "arrus/core/devices/utils.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterImplBase.h"
#include "arrus/core/devices/probe/ProbeImplBase.h"
#include "arrus/core/devices/us4r/hv/HV256Impl.h"
#include "arrus/core/devices/us4r/Us4RBuffer.h"

namespace arrus::devices {

class Us4RImpl : public Us4R {
public:
    using Us4OEMs = std::vector<Us4OEMImplBase::Handle>;

    enum class State {
        STARTED, STOPPED
    };

    ~Us4RImpl() override;

    Us4RImpl(const DeviceId &id, Us4OEMs us4oems,  std::optional<HV256Impl::Handle> hv,
             std::vector<unsigned short> channelsMask)
        : Us4R(id), us4oems(std::move(us4oems)), hv(std::move(hv)), channelsMask(std::move(channelsMask)) {}

    Us4RImpl(const DeviceId &id,
             Us4OEMs us4oems,
             ProbeAdapterImplBase::Handle &probeAdapter,
             ProbeImplBase::Handle &probe,
             std::optional<HV256Impl::Handle> hv,
             std::vector<unsigned short> channelsMask);

    Us4RImpl(Us4RImpl const &) = delete;

    Us4RImpl(Us4RImpl const &&) = delete;

    Device::RawHandle getDevice(const std::string &path) override {
        auto[root, tail] = getPathRoot(path);
        boost::algorithm::trim(root);
        boost::algorithm::trim(tail);
        if(!tail.empty()) {
            throw IllegalArgumentException(
                arrus::format(
                    "Us4R devices allows access only to the top-level "
                    "devices (got relative path: '{}')", path)
            );
        }
        DeviceId componentId = DeviceId::parse(root);
        return getDevice(componentId);
    }

    Device::RawHandle getDevice(const DeviceId &deviceId) {
        auto ordinal = deviceId.getOrdinal();
        switch(deviceId.getDeviceType()) {
            case DeviceType::Us4OEM:
                return getUs4OEM(ordinal);
            case DeviceType::ProbeAdapter:
                return getProbeAdapter(ordinal);
            case DeviceType::Probe:
                return getProbe(ordinal);
            default:
                throw DeviceNotFoundException(deviceId);
        }
    }

    Us4OEM::RawHandle getUs4OEM(Ordinal ordinal) override {
        if(ordinal >= us4oems.size()) {
            throw DeviceNotFoundException(
                DeviceId(DeviceType::Us4OEM, ordinal));
        }
        return us4oems.at(ordinal).get();
    }


    ProbeAdapter::RawHandle getProbeAdapter(Ordinal ordinal) override {
        if(ordinal > 0 || !probeAdapter.has_value()) {
            throw DeviceNotFoundException(
                DeviceId(DeviceType::ProbeAdapter, ordinal));
        }
        return probeAdapter.value().get();
    }

    Probe::RawHandle getProbe(Ordinal ordinal) override {
        if(ordinal > 0 || !probe.has_value()) {
            throw DeviceNotFoundException(
                DeviceId(DeviceType::Probe, ordinal));
        }
        return probe.value().get();
    }

    std::pair<
        std::shared_ptr<arrus::devices::HostBuffer>,
        std::shared_ptr<arrus::devices::FrameChannelMapping>
    >
    upload(const ops::us4r::TxRxSequence &seq,
           unsigned short rxBufferNElements, unsigned short hostBufferNElements) override;

    void start() override;

    void stop() override;

    void setVoltage(Voltage voltage);

    void disableHV();

    void setTgcCurve(const std::vector<float> &tgcCurvePoints) override;

    std::vector<unsigned short> getChannelsMask() override;

    uint8_t getNumberOfUs4OEMs() override {
      return (uint8_t)(us4oems.size());
    }

private:
    std::mutex deviceStateMutex;
    Logger::Handle logger;
    Us4OEMs us4oems;
    std::optional<ProbeAdapterImplBase::Handle> probeAdapter;
    std::optional<ProbeImplBase::Handle> probe;
    std::optional<HV256Impl::Handle> hv;
    std::shared_ptr<Us4ROutputBuffer> buffer;
    State state{State::STOPPED};

    std::vector<unsigned short> channelsMask;

    UltrasoundDevice *getDefaultComponent();

    void stopDevice();

    void syncTrigger();

    std::tuple<Us4RBuffer::Handle, FrameChannelMapping::Handle>
    uploadSequence(const ops::us4r::TxRxSequence &seq, uint16_t rxBufferSize,
                   uint16_t rxBatchSize);

    ProbeImplBase::RawHandle getProbeImpl() {
        return probe.value().get();
    }
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
