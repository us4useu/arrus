#ifndef API_MATLAB_WRAPPERS_DEVICES_PROBE_LENSCONVERTER_H
#define API_MATLAB_WRAPPERS_DEVICES_PROBE_LENSCONVERTER_H

#include "api/matlab/wrappers/MexContext.h"
#include "arrus/core/api/arrus.h"
#include "api/matlab/wrappers/convert.h"

#include <mex.hpp>
#include <mexAdapter.hpp>
#include <utility>


namespace arrus::matlab::devices::probe {

using namespace ::arrus::devices;
using namespace ::arrus::matlab::converters;

class LensConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.devices.probe.Lens";

    // MATLAB object -> LensConverter.
    static LensConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        return LensConverter{
            ctx,
            ARRUS_MATLAB_GET_CPP_SCALAR(ctx, float, thickness, object),
            ARRUS_MATLAB_GET_CPP_SCALAR(ctx, float, speedOfSound, object),
            ARRUS_MATLAB_GET_CPP_OPTIONAL_SCALAR(ctx, float, focus, object),
        };
    }

    // C++ API object -> LensConverter.
    static LensConverter from(const MexContext::SharedHandle &ctx, const Lens &object) {
        return LensConverter{
            ctx,
            object.getThickness(),
            object.getSpeedOfSound(),
            object.getFocus()
        };
    }

    LensConverter(const MexContext::SharedHandle &ctx, float thickness, float speedOfSound,
                  const std::optional<float> &focus)
        : ctx(ctx), thickness(thickness), speedOfSound(speedOfSound), focus(focus) {}

    // LensConverter -> C++ API object.
    [[nodiscard]] ::arrus::devices::Lens toCore() const {
        return ::arrus::devices::Lens{thickness, speedOfSound, focus};
    }

    // LensConverter -> MATLAB API object.
    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        return ctx->createObject(
            MATLAB_FULL_NAME,
            {
                ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, float, thickness),
                ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, float, speedOfSound),
                ARRUS_MATLAB_GET_MATLAB_SCALAR_KV(ctx, float, focus)
            }
        );
    }

private:
    MexContext::SharedHandle ctx;
    float thickness;
    float speedOfSound;
    std::optional<float> focus;
};
}


#endif//API_MATLAB_WRAPPERS_DEVICES_PROBE_LENSCONVERTER_H
