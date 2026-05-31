#ifndef ARRUS_CORE_API_DEVICES_PROBE_PROBEMODEL_H
#define ARRUS_CORE_API_DEVICES_PROBE_PROBEMODEL_H

#include <optional>
#include <ostream>
#include <utility>

#include "arrus/core/api/common/Interval.h"
#include "arrus/core/api/common/Tuple.h"
#include "arrus/core/api/common/exceptions.h"
#include "arrus/core/api/common/types.h"
#include "arrus/core/api/devices/probe/Lens.h"
#include "arrus/core/api/devices/probe/MatchingLayer.h"
#include "arrus/core/api/devices/probe/ProbeModelId.h"
#include "std4us/Optional.hpp"
#include "std4us/StdInterop.hpp"

namespace arrus::devices {

/**
 * A specification of the probe model.
 *
 * lens and matchingLayer storage uses std4us::Optional for ABI stability;
 * std::optional-typed constructors and accessors are kept as inline
 * header-only backward-compatibility shims.
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
        : ProbeModel(std::move(modelId), numberOfElements, pitch, txFrequencyRange, voltageRange,
                     curvatureRadius, std4us::fromStd(lens), std4us::fromStd(matchingLayer)) {}

    ProbeModel(ProbeModelId modelId,
               const Tuple<ElementIdxType> &numberOfElements,
               const Tuple<double> &pitch,
               const Interval<float> &txFrequencyRange,
               const Interval<Voltage> &voltageRange,
               const double curvatureRadius,
               std4us::Optional<Lens> lens,
               std4us::Optional<MatchingLayer> matchingLayer)
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

    std::optional<Lens> getLens() const { return std4us::toStd(lens); }
    const std4us::Optional<Lens> &getLensNative() const { return lens; }
    /** Returns true when the lens is defined for this probe model, otherwise false. */
    bool isLensDefined() const { return lens.hasValue(); }
    /** Returns lens definition. If the lens is not defined for this probe, exception will be raised. */
    const Lens &getLensOrRaiseException() { return lens.value(); }

    std::optional<MatchingLayer> getMatchingLayer() const { return std4us::toStd(matchingLayer); }
    const std4us::Optional<MatchingLayer> &getMatchingLayerNative() const { return matchingLayer; }
    /** Returns true when the matching layer is defined for this probe model, otherwise false. */
    bool isMatchingLayerDefined() const { return matchingLayer.hasValue(); }
    /** Returns matching layer definition. If the matching layer is not defined for this probe, exception will be raised. */
    const MatchingLayer &getMatchingLayerOrRaiseException() { return matchingLayer.value(); }


private:
    ProbeModelId modelId;
    Tuple<ElementIdxType> numberOfElements;
    Tuple<double> pitch;
    Interval<float> txFrequencyRange;
    Interval<Voltage> voltageRange;
    double curvatureRadius;
    std4us::Optional<Lens> lens;
    std4us::Optional<MatchingLayer> matchingLayer;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_PROBE_PROBEMODEL_H
