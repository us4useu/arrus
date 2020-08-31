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

namespace arrus {

class Us4RImpl : public Us4R {
public:
    using Us4OEMs = std::vector<Us4OEM::Handle>;

    ~Us4RImpl() override {
        getDefaultLogger()->log(LogSeverity::DEBUG, "Destroying Us4R instance");
    }

    Us4RImpl(const DeviceId &id, Us4OEMs &us4oems)
            : Us4R(id), us4oems(std::move(us4oems)) {}

    Us4RImpl(const DeviceId &id, Us4OEMs &us4oems,
             ProbeAdapter::Handle &probeAdapter, Probe::Handle &probe)
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
                return getUs4OEM(ordinal).get();
            case DeviceType::ProbeAdapter:
                return getProbeAdapter(ordinal).get();
            case DeviceType::Probe:
                return getProbe(ordinal).get();
            default:
                throw DeviceNotFoundException(deviceId);
        }
    }

    Us4OEM::Handle &getUs4OEM(Ordinal ordinal) override {
        if(ordinal >= us4oems.size()) {
            throw DeviceNotFoundException(
                    DeviceId(DeviceType::Us4OEM, ordinal));
        }
        return us4oems.at(ordinal);
    }

    ProbeAdapter::Handle &getProbeAdapter(Ordinal ordinal) override {
        if(ordinal > 0 || !probeAdapter.has_value()) {
            throw DeviceNotFoundException(
                    DeviceId(DeviceType::ProbeAdapter, ordinal));
        }
        return probeAdapter.value();
    }

    Probe::Handle &getProbe(Ordinal ordinal) override {
        if(ordinal > 0 || !probe.has_value()) {
            throw DeviceNotFoundException(
                    DeviceId(DeviceType::Probe, ordinal));
        }
        return probe.value();
    }

private:
    Us4OEMs us4oems;
    std::optional<ProbeAdapter::Handle> probeAdapter;
    std::optional<Probe::Handle> probe;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4RIMPL_H
