#ifndef ARRUS_CORE_API_DEVICES_PROBE_LENS_H
#define ARRUS_CORE_API_DEVICES_PROBE_LENS_H

#include <optional>

namespace arrus::devices {

/**
 * The lens applied on the surface of the probe.
 *
 * Currently, the model of the lens is quite basic and accustomed mostly to
 * the linear array probes, e.g. we assume that the lens is dedicated to be
 * focusing in the elevation direction.
 *
 * Parameters:
 * - thickness: lens thickness of linear array, measured at center of the elevation,
 * - speedOfSound: the speed of sound in the lens material,
 * - focus: geometric focus (along elevation axis) measured in water
 */
class Lens {
public:
    Lens(float thickness, float speedOfSound, const std::optional<float> &focus = std::nullopt)
        : thickness(thickness), speedOfSound(speedOfSound), focus(focus) {}

    float getThickness() const { return thickness; }
    float getSpeedOfSound() const { return speedOfSound; }
    std::optional<float> getFocus() const { return focus; }

private:
    /* Lens thickness of linear array, measured at center of the elevation. */
    float thickness;
    /** The speed of sound in the lens material. */
    float speedOfSound;
    /** Geometric elevation focus measured in water */
    std::optional<float> focus;
};

}

#endif//ARRUS_CORE_API_DEVICES_PROBE_LENS_H
