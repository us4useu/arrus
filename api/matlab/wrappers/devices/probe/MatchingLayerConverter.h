#ifndef API_MATLAB_WRAPPERS_DEVICES_PROBE_MATCHINGLAYERCONVERTER_H
#define API_MATLAB_WRAPPERS_DEVICES_PROBE_MATCHINGLAYERCONVERTER_H

namespace arrus::matlab::devices::probe {

using namespace ::arrus::devices;
using namespace ::arrus::matlab::converters;

class MatchingLayerConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.devices.probe.MatchingLayer";

    // MATLAB object -> MatchingLayerConverter.
    static MatchingLayerConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        return MatchingLayerConverter{
            ctx,
            ARRUS_MATLAB_GET_CPP_SCALAR(ctx, float, thickness, object),
            ARRUS_MATLAB_GET_CPP_SCALAR(ctx, float, speedOfSound, object),
        };
    }

    // C++ API object -> MatchingLayerConverter.
    static MatchingLayerConverter from(const MexContext::SharedHandle &ctx, const MatchingLayer &object) {
        return MatchingLayerConverter{
            ctx,
            object.getThickness(),
            object.getSpeedOfSound()
        };
    }

    MatchingLayerConverter(const MexContext::SharedHandle &ctx, float thickness, float speedOfSound)
        : ctx(ctx), thickness(thickness), speedOfSound(speedOfSound) {}

    // MatchingLayerConverter -> C++ API object.
    [[nodiscard]] ::arrus::devices::MatchingLayer toCore() const {
        return ::arrus::devices::MatchingLayer{thickness, speedOfSound};
    }

    // MatchingLayerConverter -> MATLAB API object.
    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        return ctx->createObject(
            MATLAB_FULL_NAME,
            {
                ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, float, thickness),
                ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, float, speedOfSound),
            }
        );
    }

private:
    MexContext::SharedHandle ctx;
    float thickness;
    float speedOfSound;
};
}

#endif//API_MATLAB_WRAPPERS_DEVICES_PROBE_MATCHINGLAYERCONVERTER_H
