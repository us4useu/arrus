#include "HV256Impl.h"

#include <utility>

namespace arrus::devices {

HV256Impl::HV256Impl(const DeviceId &id,
                     HVModelId modelId, std::unique_ptr<IDBAR> dbarLite,
                     std::unique_ptr<IHV> hv256)
    : Device(id),
    logger(getLoggerFactory()->getLogger()),
    modelId(std::move(modelId)), dbarLite(std::move(dbarLite)), hv256(std::move(hv256)) {

    INIT_ARRUS_DEVICE_LOGGER(logger, id.toString());
}
}