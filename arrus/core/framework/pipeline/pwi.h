#ifndef CPP_EXAMPLE_PWI_SEQUENCE_H
#define CPP_EXAMPLE_PWI_SEQUENCE_H

#include <utility>
#include <vector>

#include <arrus/core/api/arrus.h>
#include "imaging/ProbeModelExt.h"
#include "imaging/Metadata.h"

namespace arrus_example_imaging {

class PwiSequence {
public:
    using Aperture = ProbeModelExt::Aperture;

    PwiSequence(std::vector<Aperture> txApertures, std::vector<Aperture> rxApertures, std::vector<float> angles,
                const arrus::ops::us4r::Pulse &pulse, float speedOfSound, float pri, const std::optional<float> &sri,
                std::pair<::arrus::uint32, arrus::uint32> sampleRange, unsigned int downsamplingFactor = 1)
        : txApertures(std::move(txApertures)), rxApertures(std::move(rxApertures)), angles(std::move(angles)),
          pulse(pulse), speedOfSound(speedOfSound), pri(pri), sri(sri), sampleRange(std::move(sampleRange)),
          downsamplingFactor(downsamplingFactor) {}

    const std::vector<float> &getAngles() const { return angles; }
    const arrus::ops::us4r::Pulse &getPulse() const { return pulse; }
    float getSpeedOfSound() const { return speedOfSound; }
    float getPri() const { return pri; }
    const std::optional<float> &getSri() const { return sri; }
    const std::pair<::arrus::uint32, arrus::uint32> &getSampleRange() const { return sampleRange; }
    unsigned int getDownsamplingFactor() const { return downsamplingFactor; }
    const std::vector<Aperture> &getTxApertures() const { return txApertures; }
    const std::vector<Aperture> &getRxApertures() const { return rxApertures; }

private:
    std::vector<Aperture> txApertures;
    std::vector<Aperture> rxApertures;
    std::vector<float> angles;
    ::arrus::ops::us4r::Pulse pulse{6e6, 2, false};
    float speedOfSound{1540};
    float pri;
    std::optional<float> sri;
    ::std::pair<::arrus::uint32, arrus::uint32> sampleRange{0, 2048};
    unsigned downsamplingFactor{1};
};

/**
 * Uploads a given PWI sequence on the devices in the session.
 */
std::tuple<std::shared_ptr<::arrus::framework::Buffer>, NdArrayDef, std::shared_ptr<Metadata>>
upload(::arrus::session::Session *session, const PwiSequence &seq, const std::vector<ProbeModelExt> &probes);

}

#endif//CPP_EXAMPLE_PWI_SEQUENCE_H
