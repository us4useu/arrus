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
#include "arrus/core/devices/us4r/Watchdog.h"
#include "arrus/core/devices/us4r/Us4ROutputBuffer.h"

namespace arrus::devices {

class Us4RImpl : public Us4R {
public:
    using Us4OEMs = std::vector<Us4OEMImplBase::Handle>;

    enum class State {
        STARTED, STOPPED
    };

    enum Mode {
        SYNC, ASYNC
    };

    ~Us4RImpl() override;

    Us4RImpl(const DeviceId &id, Us4OEMs us4oems,
             std::optional<HV256Impl::Handle> hv)
        : Us4R(id), us4oems(std::move(us4oems)),
          hv(std::move(hv)) {
    }

    Us4RImpl(const DeviceId &id,
             Us4OEMs us4oems,
             ProbeAdapterImplBase::Handle &probeAdapter,
             ProbeImplBase::Handle &probe,
             std::optional<HV256Impl::Handle> hv);

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
        std::shared_ptr<arrus::devices::FrameChannelMapping>,
        std::shared_ptr<arrus::devices::HostBuffer>
    >
    uploadSync(const ops::us4r::TxRxSequence &seq,
               unsigned short hostBufferSize,
               unsigned short rxBatchSize) override;

    std::pair<
        std::shared_ptr<arrus::devices::FrameChannelMapping>,
        std::shared_ptr<arrus::devices::HostBuffer>
    >
    uploadAsync(const ::arrus::ops::us4r::TxRxSequence &seq,
                unsigned short rxBufferSize,
                unsigned short hostBufferSize,
                float frameRepetitionInterval) override;

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
    std::unique_ptr<HostBufferWorker> hostBufferWorker;
    std::unique_ptr<Watchdog> watchdog;
    // will be used outside
    std::shared_ptr<Us4RHostBuffer> hostBuffer;
    std::shared_ptr<Us4ROutputBuffer> asyncBuffer;
    std::mutex deviceStateMutex;
    State state{State::STOPPED};
    std::optional<Mode> mode;

    UltrasoundDevice *getDefaultComponent();

    static size_t countBufferElementSize(
        const std::vector<std::vector<DataTransfer>> &transfers);

    void stopDevice(bool stopGently = true);

    bool rxDmaCallback();

    void syncTrigger();

    std::tuple<
        FrameChannelMapping::Handle,
        std::vector<std::vector<DataTransfer>>,
        float // total PRI
    >
    uploadSequence(
        const ops::us4r::TxRxSequence &seq,
        uint16_t rxBufferSize,
        uint16_t rxBatchSize,
        bool checkpoint,
        std::optional<float> frameRepetitionInterval);

    void startAsync();
    void stopAsync();

    void startSync();
    void stopSync();
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
