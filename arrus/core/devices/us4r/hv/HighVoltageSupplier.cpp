#include "HighVoltageSupplier.h"

#include <utility>

namespace arrus::devices {

HighVoltageSupplier::HighVoltageSupplier(const DeviceId &id, HVModelId modelId,
                                         std::optional<std::unique_ptr<IDBAR>> dbar,
                                         std::unique_ptr<IHV> hv)
    : Device(id),
    logger(getLoggerFactory()->getLogger()),
    modelId(std::move(modelId)),
    dbar(std::move(dbar)),
    hv(std::move(hv)) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
}
}