#ifndef ARRUS_CORE_DEVICES_ULTRASOUND_ULTRASOUNDFILE_H
#define ARRUS_CORE_DEVICES_ULTRASOUND_ULTRASOUNDFILE_H

#include "arrus/core/api/devices/ultrasound/Ultrasound.h"

namespace arrus::devices {
class UltrasoundFile : public Ultrasound {
public:
    UltrasoundFile(const std::string &filepath, const Tuple<size_t> shape);
    ~UltrasoundFile() override = default;
    std::pair<std::shared_ptr<arrus::framework::Buffer>, ::arrus::framework::Metadata>
    upload(const ops::us4r::TxRxSequence &seq, unsigned short rxBufferSize, const ops::us4r::Scheme::WorkMode &workMode,
           const framework::DataBufferSpec &hostBufferSpec) override;
    void setVoltage(Voltage voltage) override;
    void start() override;
    void stop() override;
    void trigger() override;
    float getSamplingFrequency() const override;
private:
    std::string filepath;
    Tuple<size_t> shape;
};
}


#endif//ARRUS_CORE_DEVICES_ULTRASOUND_ULTRASOUNDFILE_H
