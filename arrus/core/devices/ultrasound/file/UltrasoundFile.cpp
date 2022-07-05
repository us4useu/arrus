#include "UltrasoundFile.h"
#include "arrus/core/api/devices/ultrasound/UltrasoundFileSettings.h"

namespace arrus::devices {

UltrasoundFile::UltrasoundFile(const DeviceId &deviceId, const UltrasoundFileSettings &settings)
    : Ultrasound(deviceId), settings(settings), logger {
    // TODO logger
    // TODO logger
   // TODO load file
}

std::pair<std::shared_ptr<arrus::framework::Buffer>, ::arrus::framework::Metadata>
UltrasoundFile::upload(const ops::us4r::TxRxSequence &seq, unsigned short rxBufferSize,
                       const ops::us4r::Scheme::WorkMode &workMode, const framework::DataBufferSpec &hostBufferSpec) {
    // Buffer only implementation?
    // TODO logger
    // TODO save work mode for kind behaviour implementation
}
void UltrasoundFile::setVoltage(Voltage voltage) {
    // TODO logger
}
void UltrasoundFile::start() {
    // TODO logger
    // TODO start producer thread
}
void UltrasoundFile::stop() {
    // TODO logger
    // TODO stop produdcer thread

}
void UltrasoundFile::trigger() {
    // TODO logger
    // TODO currently not supported
}

}


