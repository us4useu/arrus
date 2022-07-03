#ifndef ARRUS_CORE_API_DEVICES_ULTRASOUND_ULTRASOUNDFILESETTINGS_H
#define ARRUS_CORE_API_DEVICES_ULTRASOUND_ULTRASOUNDFILESETTINGS_H

#include "arrus/core/api/common/Tuple.h"
#include <string>

namespace arrus::devices {

/**
 * Ultrasound file simulated mode.
 *
 * This settings allows to start simulated mode, that reads raw channel data from a given,
 * which then will be provided in a given buffer.
 */
class UltrasoundFileSettings {
public:
    UltrasoundFileSettings(const std::string &filepath, const Tuple<size_t> &shape) : filepath(filepath), shape(shape) {}

    const std::string &getFilepath() const { return filepath; }
    const Tuple<size_t> &getShape() const { return shape; }

private:
    std::string filepath;
    Tuple<size_t> shape;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_ULTRASOUND_ULTRASOUNDFILESETTINGS_H
