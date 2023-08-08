#ifndef ARRUS_CORE_API_DEVICES_FILE_H
#define ARRUS_CORE_API_DEVICES_FILE_H

#include "Ultrasound.h"

namespace arrus::devices {

class File: public Ultrasound {
public:
    using Handle = std::unique_ptr<File>;

    explicit File(const DeviceId &id): Ultrasound(id) {}

    ~File() override = default;
};

}

#endif//ARRUS_CORE_API_DEVICES_FILE_H
