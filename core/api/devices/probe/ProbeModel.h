#ifndef ARRUS_CORE_API_DEVICES_PROBE_PROBEMODEL_H
#define ARRUS_CORE_API_DEVICES_PROBE_PROBEMODEL_H

#include "arrus/core/api/common/Tuple.h"
#include "arrus/core/api/common/Interval.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/common/exceptions.h"

namespace arrus {

/**
 * A specification of the probe model.
 */
class ProbeModel {
public:
    ProbeModel(const Tuple<ChannelIdx> &numberOfElements,
               const Tuple<double> &pitch,
               const Interval<double> &txFrequencyRange)
            : numberOfElements(numberOfElements), pitch(pitch),
              txFrequencyRange(txFrequencyRange) {
        if(numberOfElements.size() != pitch.size()) {
            throw IllegalArgumentException(
                    "Number of elements and pitch should have the same size.");
        }
    }

    [[nodiscard]] const Tuple<ChannelIdx> &getNumberOfElements() const {
        return numberOfElements;
    }

    [[nodiscard]] const Tuple<double> &getPitch() const {
        return pitch;
    }

    [[nodiscard]] const Interval<double> &getTxFrequencyRange() const {
        return txFrequencyRange;
    }

private:
    Tuple<ChannelIdx> numberOfElements;
    Tuple<double> pitch;
    Interval<double> txFrequencyRange;
};

}

#endif //ARRUS_CORE_API_DEVICES_PROBE_PROBEMODEL_H
