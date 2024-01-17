#ifndef ARRUS_CORE_DEVICES_US4R_MAPPING_ADATERTOUS4OEMMAPPINGCONVERTER_H
#define ARRUS_CORE_DEVICES_US4R_MAPPING_ADATERTOUS4OEMMAPPINGCONVERTER_H

#include <vector>

#include "arrus/core/devices/TxRxParameters.h"

namespace arrus::devices {

class AdapterToUs4OEMMappingConverter {
public:
    using Us4OEMSequences = std::vector<us4r::TxRxParametersSequence>;

    Us4OEMSequences convert(const us4r::TxRxParametersSequence &sequence) {

    }
};

}

#endif//ARRUS_CORE_DEVICES_US4R_MAPPING_ADATERTOUS4OEMMAPPINGCONVERTER_H
