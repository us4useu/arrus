#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTRXSETTINGS_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTRXSETTINGS_H

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/asserts.h"
#include "api/matlab/wrappers/convert.h"
#include "arrus/core/api/devices/us4r/RxSettings.h"

namespace arrus::matlab {

::arrus::devices::RxSettings convertToRxSettings(const MexContext::SharedHandle &ctx,
                                                 const ::matlab::data::Array &object) {
    using namespace arrus::devices;

    std::optional<uint16> dtgcAttenuation = getOptionalIntScalar<uint16>(ctx, object, "dtgcAttenuation");

    auto pgaGain = getIntScalar<uint16>(ctx, object, "pgaGain");
    auto lnaGain = getIntScalar<uint16>(ctx, object, "lnaGain");
    auto lpfCutoff = getIntScalar<uint32>(ctx, object, "lpfCutoff");

    auto activeTermination = getOptionalIntScalar<uint16>(ctx, object, "activeTermination");
    auto tgcSamples = getVector<TGCSampleValue>(ctx, object, "tgcSamples");

    return RxSettings(dtgcAttenuation, pgaGain, lnaGain, tgcSamples, lpfCutoff, activeTermination);
}
}// namespace arrus::matlab

#endif//ARRUS_API_MATLAB_WRAPPERS_DEVICES_US4R_CONVERTRXSETTINGS_H
