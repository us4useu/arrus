#ifndef ARRUS_CORE_API_DEVICES_PROBE_PROBEMODEL_H
#define ARRUS_CORE_API_DEVICES_PROBE_PROBEMODEL_H

#include <utility>
#include <ostream>

#include "arrus/core/api/common/Tuple.h"
#include "arrus/core/api/common/Interval.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/devices/probe/ProbeModelId.h"
#include "arrus/core/api/devices/probe/Lens.h"
#include "arrus/core/api/devices/probe/MatchingLayer.h"

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
               // Float, because carrier frequency can be set only to specific values
               const Interval<float> &txFrequencyRange,
               const Interval<Voltage> &voltageRange,
               const double curvatureRadius,
               std::optional<Lens> lens = std::nullopt,
               std::optional<MatchingLayer> matchingLayer = std::nullopt
               )
        : modelId(std::move(modelId)), numberOfElements(numberOfElements),
          pitch(pitch), txFrequencyRange(txFrequencyRange), voltageRange(voltageRange),
          curvatureRadius(curvatureRadius),
          lens(std::move(lens)),
          matchingLayer(std::move(matchingLayer)) {

        if(numberOfElements.size() != pitch.size()) {
            throw IllegalArgumentException(
                "Number of elements and pitch should have the same size.");
        }
        size_t probeOrder = numberOfElements.size();
        if(probeOrder != 1 && probeOrder != 2) {
            throw IllegalArgumentException("Only 1D and 2D array probes are supported");
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

    double getCurvatureRadius() const {
        return curvatureRadius;
    }

    const std::optional<Lens> &getLens() const { return lens; }
    const std::optional<MatchingLayer> &getMatchingLayer() const { return matchingLayer; }

private:
    ProbeModelId modelId;
    Tuple<ElementIdxType> numberOfElements;
    Tuple<double> pitch;
    Interval<float> txFrequencyRange;
    Interval<Voltage> voltageRange;
    double curvatureRadius;
    std::optional<Lens> lens;
    std::optional<MatchingLayer> matchingLayer;
};

}

#endif //ARRUS_CORE_API_DEVICES_PROBE_PROBEMODEL_H
