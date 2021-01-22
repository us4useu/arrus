#ifndef ARRUS_API_MATLAB_WRAPPERS_SESSION_CONVERTSESSIONSETTINGS_H
#define ARRUS_API_MATLAB_WRAPPERS_SESSION_CONVERTSESSIONSETTINGS_H

#include "arrus/api/matlab/wrappers/MexContext.h"
#include "arrus/core/api/session/SessionSettings.h"
#include "arrus/api/matlab/wrappers/devices/us4r/convertUs4RSettings.h"
#include "mex.hpp"

namespace arrus::matlab {
    arrus::session::SessionSettings
    convertToSessionSettings(const MexContext::SharedHandle &ctx,
                             const ::matlab::data::Array &object) {
        auto matlabEngine = ctx->getMatlabEngine();

        // us4RSettings
        ::matlab::data::Array us4rSettingsObj = matlabEngine->getProperty(
            object, "us4RSettings");
        ::arrus::devices::Us4RSettings us4RSettings =
            convertToUs4RSettings(ctx, us4rSettingsObj);
        return ::arrus::session::SessionSettings(us4RSettings);
    }

}

#endif //ARRUS_API_MATLAB_WRAPPERS_SESSION_CONVERTSESSIONSETTINGS_H
