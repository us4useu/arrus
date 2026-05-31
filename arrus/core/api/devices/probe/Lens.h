#ifndef ARRUS_CORE_API_DEVICES_PROBE_LENS_H
#define ARRUS_CORE_API_DEVICES_PROBE_LENS_H

#include <optional>

#include "std4us/Optional.hpp"
#include "std4us/StdInterop.hpp"

namespace arrus::devices {

/**
 * The lens applied on the surface of the probe.
 *
 * Currently, the model of the lens is quite basic and accustomed mostly to
 * the linear array probes, e.g. we assume that the lens is dedicated to be
 * focusing in the elevation direction.
 *
 * Parameters:
 * - thickness: lens thickness measured at center of the elevation,
 * - speedOfSound: the speed of sound in the lens material,
 * - focus: geometric elevation focus in water
 *
 * focus storage uses std4us::Optional for ABI stability. The std::optional
 * constructor / accessor are inline header-only backward-compatibility shims.
 */
class Lens {
public:
    Lens(float thickness, float speedOfSound, const std::optional<float> &focus = std::nullopt)
        : thickness(thickness), speedOfSound(speedOfSound), focus(std4us::fromStd(focus)) {}

    Lens(float thickness, float speedOfSound, std4us::Optional<float> focus)
        : thickness(thickness), speedOfSound(speedOfSound), focus(std::move(focus)) {}

    float getThickness() const { return thickness; }
    float getSpeedOfSound() const { return speedOfSound; }

    std::optional<float> getFocus() const { return std4us::toStd(focus); }

    const std4us::Optional<float> &getFocusNative() const { return focus; }

private:
    /* Lens thickness measured at center of the elevation. */
    float thickness;
    /** The speed of sound in the lens material. */
    float speedOfSound;
    /** Geometric elevation focus in water */
    std4us::Optional<float> focus;
};

}// namespace arrus::devices

#endif//ARRUS_CORE_API_DEVICES_PROBE_LENS_H
