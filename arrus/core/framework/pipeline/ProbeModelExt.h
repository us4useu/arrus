#ifndef CPP_EXAMPLE_IMAGING_PROBEMODELEXT_H
#define CPP_EXAMPLE_IMAGING_PROBEMODELEXT_H

#include <arrus/core/api/arrus.h>

#include "imaging/NdArray.h"
#include <utility>

namespace arrus_example_imaging {
/**
 * An adapter to arrus::devices::ProbeModel class. Note: the functions available in this
 * class will be moved in some time to the arrus::devices::ProbeModelExt implementation.
 *
 * Note: only 1D arrus probes are supported here.
 *
 * The center of coordinate system is located in the center of probe.
 *
 * @param pitch: distance between two consecutive elements
 * @param curvatureRadius: probe's curvature radius, inf means a flat probe
 * @param axis: axis along which probe elements are located.
 */
class ProbeModelExt {
public:
    class Aperture {
    public:
        using ChannelsMask = std::vector<bool>;
        Aperture(uint16_t ordinal, const ChannelsMask &mask) : ordinal(ordinal), mask(mask) {}
        uint16_t getOrdinal() const { return ordinal; }
        const ChannelsMask &getMask() const { return mask; }

    private:
        uint16_t ordinal;
        ChannelsMask mask;
    };

    enum class Axis { OX, OY };

    ProbeModelExt(uint16_t ordinal, arrus::devices::ProbeModel probe, uint16_t startChannel, uint16_t stopChannel,
                  float pitch, float curvatureRadius = std::numeric_limits<float>::infinity(), Axis axis = Axis::OX)
        : ordinal(ordinal), arrusProbe(std::move(probe)), startChannel(startChannel), stopChannel(stopChannel),
          pitch(pitch), curvatureRadius(curvatureRadius), axis(axis) {
        if(startChannel >= stopChannel) {
            throw std::runtime_error("Start channel should be less than stop channel: "
                                     + std::to_string(startChannel) + " vs. "
                                     + std::to_string(startChannel));
        }
        if(stopChannel > arrusProbe.getNumberOfElements()[0]) {
            throw std::runtime_error("The stop channel parameter should not be greater than arrus "
                                     "probe definition size: "
                                     + std::to_string(stopChannel) + " vs. "
                                     + std::to_string(arrusProbe.getNumberOfElements()[0]));
        }

        auto nElements = stopChannel-startChannel;

        NdArray elementPosition = NdArray::arange(-((float)nElements-1)/2.0f, ((float)nElements/2.0f))*pitch;
        if(curvatureRadius == std::numeric_limits<float>::infinity()) {
            // Flat array.
            setLateralPosition(elementPosition);
            this->z = NdArray::zeros<float>(nElements);
            this->angle = NdArray::zeros<float>(nElements);
        }
        else {
            this->angle = elementPosition/curvatureRadius;
            setLateralPosition(this->angle.sin()*curvatureRadius);
            this->z = this->angle.cos()*curvatureRadius;
            this->z = this->z-this->z.min<float>();
        }
        fullAperture = std::vector<bool>(arrusProbe.getNumberOfElements()[0], false);

        std::fill(std::begin(fullAperture)+startChannel, std::begin(fullAperture)+stopChannel, true);
    }

    /**
     * Returns position of the center of probe OX
     */
    const NdArray &getElementPositionX() const { return x; }

    /**
     * Returns position of the center of probe OY
     */
    const NdArray &getElementPositionY() const { return y; }

    const NdArray &getElementPositionLateral() const {
        if(axis == Axis::OX) {
            return getElementPositionX();
        }
        else {
            return getElementPositionY();
        }
    }

    /**
     * Return the position of the center of probe OZ
     */
    const NdArray &getElementPositionZ() const { return z; }

    const NdArray &getElementAngle() const {return angle; }

    /**
     * Returns the mask of all the channels assigned to the given probe.
     */
    Aperture getFullAperture() const { return Aperture(ordinal, fullAperture); }

    float getCurvatureRadius() const { return curvatureRadius; }

    /**
     * @return true if probe elements are located on OX axis (i.e. OY coordinate is always 0), false otherwise
     */
    bool isOX() const {
        return axis == Axis::OX;
    }

    /**
     * @return true if probe elements are located on OY axis (i.e. OX coordinate is always 0), false otherwise
     */
    bool isOY() const {
        return axis == Axis::OY;
    }

    size_t getNumberOfElements() const {
        return stopChannel-startChannel;
    }

    uint16_t getStartChannel() const { return startChannel; }
    uint16_t getStopChannel() const { return stopChannel; }

private:

    void setLateralPosition(const NdArray& position) {
        if(this->axis == Axis::OX) {
            this->x = position;
            this->y = NdArray::zeros<float>(position.getNumberOfElements());
        }
        else {
            this->y = position;
            this->x = NdArray::zeros<float>(position.getNumberOfElements());
        }
    }


    uint16_t ordinal;
    arrus::devices::ProbeModel arrusProbe;
    uint16_t startChannel, stopChannel;
    std::vector<bool> fullAperture;
    NdArray x, y, z, angle;
    float curvatureRadius;
    float pitch;
    Axis axis;
};
}// namespace arrus_example_imaging

#endif//CPP_EXAMPLE_IMAGING_PROBEMODELEXT_H
