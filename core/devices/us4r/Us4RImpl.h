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
#include "arrus/core/devices/us4r/RxBuffer.h"
#include "arrus/core/devices/us4r/HostBufferWorker.h"
#include "arrus/core/devices/us4r/Us4RHostBuffer.h"

namespace arrus::devices {

class Us4RImpl : public Us4R {
public:
    using Us4OEMs = std::vector<Us4OEMImplBase::Handle>;

    enum class State {STARTED, STOPPED};

    ~Us4RImpl() override;

    Us4RImpl(const DeviceId &id, Us4OEMs us4oems,
             std::optional<HV256Impl::Handle> hv)
        : Us4R(id),  us4oems(std::move(us4oems)),
          hv(std::move(hv)) {
    }

    Us4RImpl(const DeviceId &id,
             Us4OEMs us4oems,
             ProbeAdapterImplBase::Handle &probeAdapter,
             ProbeImplBase::Handle &probe,
             std::optional<HV256Impl::Handle> hv);

    Us4RImpl(Us4RImpl const&) = delete;
    Us4RImpl(Us4RImpl const&&) = delete;

    Device::RawHandle getDevice(const std::string &path) override {
        auto[root, tail] = getPathRoot(path);
        boost::algorithm::trim(root);
        boost::algorithm::trim(tail);
        if(!tail.empty()) {
            throw IllegalArgumentException(
                arrus::format("Us4R devices allows access onl to the top-level "
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

    std::tuple<
        FrameChannelMapping::Handle,
        HostBuffer::SharedHandle
    >
    upload(const ops::us4r::TxRxSequence &us4oemOrdinal) override;

    void start() override;

    void stop() override;

    void setVoltage(Voltage voltage);

    void disableHV();

private:
    Logger::Handle logger;
    Us4OEMs us4oems;
    std::optional<ProbeAdapterImplBase::Handle> probeAdapter;
    std::optional<ProbeImplBase::Handle> probe;
    std::optional<HV256Impl::Handle> hv;
    std::shared_ptr<RxBuffer> currentRxBuffer;
    std::unique_ptr<HostBufferWorker> dataCarrier;
    // will be used outside
    std::shared_ptr<Us4RHostBuffer> hostBuffer;
    State state{State::STOPPED};

    UltrasoundDevice *getDefaultComponent();

    static size_t countBufferElementSize(const std::vector<std::vector<DataTransfer>>& transfers);

    void stopDevice();
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
