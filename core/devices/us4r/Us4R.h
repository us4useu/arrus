#ifndef ARRUS_CORE_DEVICES_US4R_US4R_H
#define ARRUS_CORE_DEVICES_US4R_US4R_H

#include <memory>

#include "core/devices/Device.h"
#include "core/devices/us4oem/Us4OEM.h"
#include "core/devices/adapter/Adapter.h"
#include "core/devices/probe/Probe.h"


namespace arrus {

/**
 * Us4R device system.
 */
class Us4R : public Device {
public:
    using Handle = std::unique_ptr<Us4R>;

    explicit Us4R(const DeviceId &id): Device(id) {}

    virtual ~Us4R() = default;

    /**
     * Returns a handle to Us4OEM identified by given ordinal number.
     *
     * @param ordinal ordinal number of the us4oem to get
     * @return a handle to the us4oem module
     */
    virtual Us4OEM::Handle &getUs4OEM(Ordinal ordinal) = 0;

    /**
     * Returns a handle to an adapter identified by given ordinal number.
     *
     * @param ordinal ordinal number of the adapter to get
     * @return a handle to the adapter device
     */
    virtual Adapter::Handle &getAdapter(Ordinal ordinal) = 0;

    /**
     * Returns a handle to a probe identified by given ordinal number.
     *
     * @param ordinal ordinal number of the probe to get
     * @return a handle to the probe
     */
    virtual Probe::Handle &getProbe(Ordinal ordinal) = 0;

    Us4R(Us4R const&) = delete;
    Us4R(Us4R const&&) = delete;
    void operator=(Us4R const&) = delete;
    void operator=(Us4R const&&) = delete;
};

}

#endif //ARRUS_CORE_DEVICES_US4R_US4R_H
