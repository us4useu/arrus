#ifndef ARRUS_CORE_API_DEVICES_PROBE_MATCHINGLAYER_H
#define ARRUS_CORE_API_DEVICES_PROBE_MATCHINGLAYER_H

namespace arrus::devices {

/**
 * The matching layer between the lens and probe.
 *
 * Parameters:
 * - thickness: lens thickness of linear array, measured at center of the elevation,
 * - speedOfSound: the speed of sound in the lens material,
 */

class MatchingLayer {
public:
    MatchingLayer(float thickness, float speedOfSound) : thickness(thickness), speedOfSound(speedOfSound) {}
    float getThickness() const { return thickness; }
    float getSpeedOfSound() const { return speedOfSound; }

private:
    float thickness;
    float speedOfSound;
};

}

#endif//ARRUS_CORE_API_DEVICES_PROBE_MATCHINGLAYER_H
