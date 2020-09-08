#include "ProbeModel.h"

#include "arrus/common/format.h"


namespace arrus {
std::ostream &operator<<(std::ostream &os, const ProbeModel &model) {
    os << "modelId: " << model.getModelId().getName() << ", "
       << model.getModelId().getManufacturer()
       << " numberOfElements: "
       << toString(model.getNumberOfElements())
       << " pitch: " << toString(model.getPitch())
       << " txFrequencyRange: " << toString(model.getTxFrequencyRange());
    return os;
}
}

