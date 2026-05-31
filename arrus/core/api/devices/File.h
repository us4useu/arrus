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
    void triggerNative(bool sync, std4us::Optional<long long> timeout) override = 0;
    void syncNative(std4us::Optional<long long> timeout) override = 0;
    float getSamplingFrequency() const override = 0;
    float getCurrentSamplingFrequency() const override = 0;
    arrus::devices::Probe *getProbe(Ordinal ordinal) override = 0;

    std::pair<std::shared_ptr<framework::Buffer>, std::vector<std::shared_ptr<session::Metadata>>>
    setSubsequences(const std::vector<Slice> &slices, const std::vector<std::optional<float>> &sris) override = 0;
};

}

#endif//ARRUS_CORE_API_DEVICES_FILE_H
