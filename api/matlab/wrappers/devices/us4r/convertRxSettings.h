#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTRXSETTINGS_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTRXSETTINGS_H

#include "arrus/api/matlab/wrappers/MexContext.h"
#include "arrus/api/matlab/wrappers/convert.h"
#include "arrus/api/matlab/wrappers/asserts.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"

namespace arrus::matlab {

::arrus::devices::RxSettings
convertToRxSettings(const MexContext::SharedHandle &ctx,
                    const ::matlab::data::Array &object) {
    using namespace arrus::devices;

    std::optional<uint16> dtgcAttenuation = getOptionalIntScalar<uint16>(
        ctx, object, "dtgcAttenuation");

    auto pgaGain = getIntScalar<uint16>(ctx, object, "pgaGain");
    auto lnaGain = getIntScalar<uint16>(ctx, object, "lnaGain");
    auto lpfCutoff = getIntScalar<uint32>(ctx, object, "lpfCutoff");

    auto activeTermination = getOptionalIntScalar<uint16>(ctx, object,
                                                          "activeTermination");
    auto tgcSamples = getVector<TGCSampleValue>(ctx, object, "tgcSamples");

    return RxSettings(dtgcAttenuation, pgaGain, lnaGain, tgcSamples,
                      lpfCutoff, activeTermination);
}
}

#endif //ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTRXSETTINGS_H
