#ifndef ARRUS_CORE_API_DEVICES_PROBE_MATCHINGLAYER_H
#define ARRUS_CORE_API_DEVICES_PROBE_MATCHINGLAYER_H

namespace arrus::devices {

/**
 * The matching layer applied directly on the probe elements.
 *
 * Parameters:
 * - thickness: matching layer thickness,
 * - speedOfSound: matching layer speed of sound,
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
