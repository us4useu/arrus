#include "ProbeModel.h"

namespace arrus::devices {
std::ostream &operator<<(std::ostream &os, const ProbeModel &model) {
    os << "modelId: " << model.getModelId().getName() << ", "
       << model.getModelId().getManufacturer()
       << " numberOfElements: "
       << model.getNumberOfElements().toString()
       << " pitch: " << model.getPitch().toString()
       << " txFrequencyRange: " << model.getTxFrequencyRange().toString();
    return os;
}
}

