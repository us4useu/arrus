#ifndef ARRUS_CORE_API_DEVICES_CPU_H
#define ARRUS_CORE_API_DEVICES_CPU_H

#include "Device.h"

namespace arrus::devices {

class Cpu : Device {
    using Handle = std::unique_ptr<Cpu>;
    explicit Cpu(const DeviceId &id): Device(id) {}
    ~Cpu() override = default;
};

}


#endif//ARRUS_CORE_API_DEVICES_CPU_H
