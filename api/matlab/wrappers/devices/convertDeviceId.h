#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_CONVERTDEVICEID_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_CONVERTDEVICEID_H

#include "mex.hpp"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/api/matlab/wrappers/convert.h"
#include "arrus/api/matlab/wrappers/MexContext.h"

namespace arrus::matlab {

    DeviceId convertToDeviceId(const MexContext::SharedHandle& ctx,
                               const ::matlab::data::Array &object) {
        auto matlabEngine = ctx->getMatlabEngine();
        std::string deviceTypeStr = convertToString(
            matlabEngine->getProperty(object, "deviceType"));
        DeviceType dt = arrus::parseToDeviceTypeEnum(deviceTypeStr);

        ::matlab::data::TypedArray<double> ordinalArr =
            matlabEngine->getProperty(object, "ordinal");
        ARRUS_MATLAB_REQUIRES_SCALAR(ordinalArr,
                                     "Device ordinal value should be a scalar");
        double ordinal = ordinalArr[0];
        ARRUS_MATLAB_REQUIRES_DATA_TYPE_VALUE(ordinal, Ordinal);
        return DeviceId(dt, (Ordinal)ordinal);
    }

}


#endif //ARRUS_API_MATLAB_WRAPPERS_DEVICES_CONVERTDEVICEID_H
