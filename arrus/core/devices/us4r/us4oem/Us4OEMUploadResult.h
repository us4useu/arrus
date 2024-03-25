#ifndef ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEM_UPLOAD_RESULT_H
#define ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEM_UPLOAD_RESULT_H
#include "Us4OEMBuffer.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"

#include <unordered_map>
#include <utility>

namespace arrus::devices {
/**
 * Us4OEM upload method result.
 * The result informs the caller where us4OEM data will be stored in the device RAM,
 * and how to process the data back to logical, us4OEM's, aperture (FCM).
 * This information can be used e.g. to determine data transfers from us4OEM to host memory.
 */
class Us4OEMUploadResult {
public:
    Us4OEMUploadResult(Us4OEMBuffer bufferDescription, std::vector<FrameChannelMapping::Handle> fcms)
        : bufferDescription(std::move(bufferDescription)), fcms(std::move(fcms)) {}

    [[nodiscard]] const Us4OEMBuffer &getBufferDescription() const { return bufferDescription; }
    FrameChannelMapping::RawHandle getFCM(size_t sequenceId) { return fcms.at(sequenceId).get(); }
    /** NOTE: THIS FUNCTIONS MAKES this->fcms no more usable !!! */
    std::vector<FrameChannelMapping::Handle> acquireFCMs() {return std::move(fcms); }
private:
    Us4OEMBuffer bufferDescription;
    // Sequence id -> FCM.
    std::vector<FrameChannelMapping::Handle> fcms;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_DEVICES_US4R_US4OEM_US4OEM_UPLOAD_RESULT_H
