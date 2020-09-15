#ifndef ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
#define ARRUS_CORE_DEVICES_US4R_US4RIMPL_H

#include <unordered_map>
#include <utility>

#include <boost/algorithm/string.hpp>

#include "arrus/core/devices/utils.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/devices/us4r/Us4R.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/devices/us4r/us4oem/Us4OEMImpl.h"
#include "arrus/core/devices/us4r/probeadapter/ProbeAdapterImpl.h"
#include "arrus/core/devices/probe/ProbeImpl.h"

namespace arrus::devices {

class Us4RImpl : public Us4R {
public:
    using Us4OEMs = std::vector<Us4OEMImpl::Handle>;

    ~Us4RImpl() override {
        getDefaultLogger()->log(LogSeverity::DEBUG, "Destroying Us4R instance");
    }

    Us4RImpl(const DeviceId &id, Us4OEMs &us4oems)
            : Us4R(id), us4oems(std::move(us4oems)) {}

    Us4RImpl(const DeviceId &id,
             Us4OEMs &us4oems,
             ProbeAdapterImpl::Handle &probeAdapter,
             ProbeImpl::Handle &probe)
            : Us4R(id), us4oems(std::move(us4oems)),
              probeAdapter(std::move(probeAdapter)),
              probe(std::move(probe)) {}

    Device::RawHandle getDevice(const std::string &path) override {
        auto [root, tail] = getPathRoot(path);
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

private:
    Us4OEMs us4oems;
    std::optional<ProbeAdapterImpl::Handle> probeAdapter;
    std::optional<ProbeImpl::Handle> probe;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
