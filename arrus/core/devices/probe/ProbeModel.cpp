#include "ProbeModel.h"

#include "arrus/common/format.h"


namespace arrus::devices {
std::ostream &operator<<(std::ostream &os, const ProbeModel &model) {
    os << "modelId: " << model.getModelId().getName() << ", "
       << model.getModelId().getManufacturer()
       << " numberOfElements: "
       << ::arrus::toString(model.getNumberOfElements())
       << " pitch: " << ::arrus::toString(model.getPitch())
       << " txFrequencyRange: " << ::arrus::toString(model.getTxFrequencyRange());
    return os;
}
}

