#ifndef ARRUS_CORE_API_DEVICES_FILE_H
#define ARRUS_CORE_API_DEVICES_FILE_H

#include "Ultrasound.h"

namespace arrus::devices {

class File: public Ultrasound {
public:
    using Handle = std::unique_ptr<File>;

    explicit File(const DeviceId &id): Ultrasound(id) {}
    ~File() override = default;
    using Device::getDeviceId; // required by SWIG wrapper

    std::pair<framework::Buffer::SharedHandle, std::vector<session::Metadata::SharedHandle>>
    upload(const ops::us4r::Scheme &scheme) override = 0;
    void start() override = 0;
    void stop() override = 0;
    void trigger() override = 0;
    float getSamplingFrequency() const override = 0;
    float getCurrentSamplingFrequency() const override = 0;
    ProbeModel getProbeModel(Ordinal ordinal) override = 0;
};

}

#endif//ARRUS_CORE_API_DEVICES_FILE_H
