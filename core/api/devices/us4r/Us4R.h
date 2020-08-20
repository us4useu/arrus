#ifndef ARRUS_CORE_DEVICES_US4R_US4R_H
#define ARRUS_CORE_DEVICES_US4R_US4R_H

#include <memory>

#include "arrus/core/api/devices/Device.h"
#include "arrus/core/api/devices/DeviceWithComponents.h"
#include "arrus/core/api/devices/us4r/Us4OEM.h"
#include "arrus/core/api/devices/us4r/ProbeAdapter.h"
#include "arrus/core/api/devices/probe/Probe.h"


namespace arrus {

/**
 * Us4R system: a group of Us4OEM modules and related components.
 */
class Us4R : public DeviceWithComponents {
public:
    using Handle = std::unique_ptr<Us4R>;

    explicit Us4R(const DeviceId &id): DeviceWithComponents(id) {}

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
    virtual ProbeAdapter::Handle &getProbeAdapter(Ordinal ordinal) = 0;

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
