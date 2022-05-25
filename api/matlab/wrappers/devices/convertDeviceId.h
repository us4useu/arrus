#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_CONVERTDEVICEID_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_CONVERTDEVICEID_H

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/convert.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "mex.hpp"

namespace arrus::matlab {

::arrus::devices::DeviceId convertToDeviceId(const MexContext::SharedHandle &ctx, const ::matlab::data::Array &object) {
    using namespace arrus::devices;

    auto matlabEngine = ctx->getMatlabEngine();
    std::string deviceTypeStr = convertToString(matlabEngine->getProperty(object, "deviceType"));
    DeviceType dt = arrus::devices::parseToDeviceTypeEnum(deviceTypeStr);

    ::matlab::data::TypedArray<double> ordinalArr = matlabEngine->getProperty(object, "ordinal");
    ARRUS_MATLAB_REQUIRES_SCALAR(ordinalArr, "Device ordinal value should be a scalar");
    double ordinal = ordinalArr[0];
    ARRUS_MATLAB_REQUIRES_DATA_TYPE_VALUE(ordinal, Ordinal);
    ARRUS_MATLAB_REQUIRES_INTEGER(ordinal);
    return DeviceId(dt, (Ordinal) ordinal);
}

}// namespace arrus::matlab

#endif//ARRUS_API_MATLAB_WRAPPERS_DEVICES_CONVERTDEVICEID_H
