#ifndef ARRUS_CORE_API_DEVICES_PROBE_PROBEMODEL_H
#define ARRUS_CORE_API_DEVICES_PROBE_PROBEMODEL_H

#include <utility>
#include <ostream>

#include "arrus/core/api/common/Tuple.h"
#include "arrus/core/api/common/Interval.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/devices/probe/ProbeModelId.h"

namespace arrus::devices {

/**
 * A specification of the probe model.
 */
class ProbeModel {
public:

    using ElementIdxType = ChannelIdx;

    ProbeModel(ProbeModelId modelId,
               const Tuple<ElementIdxType> &numberOfElements,
               const Tuple<double> &pitch,
               const Interval<float> &txFrequencyRange,
               const Interval<Voltage> &voltageRange)
        : modelId(std::move(modelId)), numberOfElements(numberOfElements),
          pitch(pitch), txFrequencyRange(txFrequencyRange), voltageRange(voltageRange) {

        if(numberOfElements.size() != pitch.size()) {
            throw IllegalArgumentException(
                "Number of elements and pitch should have the same size.");
        }
    }

    const ProbeModelId &getModelId() const {
        return modelId;
    }

    const Tuple<ElementIdxType> &getNumberOfElements() const {
        return numberOfElements;
    }

    const Tuple<double> &getPitch() const {
        return pitch;
    }

    const Interval<float> &getTxFrequencyRange() const {
        return txFrequencyRange;
    }

    const Interval<Voltage> &getVoltageRange() const {
        return voltageRange;
    }

private:
    ProbeModelId modelId;
    Tuple<ElementIdxType> numberOfElements;
    Tuple<double> pitch;
    Interval<float> txFrequencyRange;
    Interval<Voltage> voltageRange;
};

}

#endif //ARRUS_CORE_API_DEVICES_PROBE_PROBEMODEL_H
