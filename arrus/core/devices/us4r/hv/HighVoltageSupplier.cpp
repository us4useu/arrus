#include "HighVoltageSupplier.h"

#include <utility>

namespace arrus::devices {

HighVoltageSupplier::HighVoltageSupplier(const DeviceId &id, HVModelId modelId)
    : Device(id),
    logger(getLoggerFactory()->getLogger()),
    modelId(std::move(modelId)) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
}
}