#ifndef ARRUS_CORE_DEVICES_US4R_MAPPING_US4OEMSUBAPERTUREGENERATOR_H
#define ARRUS_CORE_DEVICES_US4R_MAPPING_US4OEMSUBAPERTUREGENERATOR_H

namespace arrus::devices {

class Us4OEMSubapertureGenerator {
public:
    using Us4OEMSequences = std::vector<us4r::TxRxParametersSequence>;

    Us4OEMSequences convert(const Us4OEMSequences &sequences) {
    }
private:
};
}


#endif//ARRUS_CORE_DEVICES_US4R_MAPPING_US4OEMSUBAPERTUREGENERATOR_H
