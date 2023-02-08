#ifndef ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_PROBE_MODEL_ID_H
#define ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_PROBE_MODEL_ID_H

#include "api/matlab/wrappers/MexContext.h"
#include "arrus/core/api/arrus.h"
#include "api/matlab/wrappers/convert.h"

#include "arrus/core/api/devices/probe/ProbeModelId.h"
#include <mex.hpp>
#include <mexAdapter.hpp>

namespace arrus::matlab::devices::probe {

using namespace ::arrus::devices;
using namespace ::arrus::matlab::converters;


class ProbeModelIdConverter {
public:
    inline static const std::string MATLAB_FULL_NAME = "arrus.devices.probe.ProbeModelId";

    // MATLAB object -> ProbeModelIdConverter.
    static ProbeModelIdConverter from(const MexContext::SharedHandle &ctx, const MatlabElementRef &object) {
        return ProbeModelIdConverter{
            ctx,
            ARRUS_MATLAB_GET_CPP_REQUIRED_SCALAR(ctx, std::string, manufacturer, object),
            ARRUS_MATLAB_GET_CPP_REQUIRED_SCALAR(ctx, std::string, name, object),
        };
    }

    // C++ API object -> ProbeModelIdConverter.
    static ProbeModelIdConverter from(const MexContext::SharedHandle &ctx, const ProbeModelId &object) {
        return ProbeModelIdConverter{ctx, object.getManufacturer(), object.getName()};
    }

    ProbeModelIdConverter(MexContext::SharedHandle ctx, std::string manufacturer, std::string name)
        : ctx(std::move(ctx)), manufacturer(std::move(manufacturer)), name(std::move(name)){}

    // ProbeModelIdConverter -> C++ API object.
    [[nodiscard]] ::arrus::devices::ProbeModelId toCore() const {
        return ::arrus::devices::ProbeModelId{manufacturer, name};
    }

    // ProbeModelIdConverter -> MATLAB API object.
    [[nodiscard]] ::matlab::data::Array toMatlab() const {
        // TODO
        return ctx->createObject(
            MATLAB_FULL_NAME,
            {
                ARRUS_MATLAB_GET_MATLAB_STRING_KV_EXPLICIT(ctx, u"manufacturer", manufacturer),
                ARRUS_MATLAB_GET_MATLAB_STRING_KV_EXPLICIT(ctx, u"name", name),
            }
        );
    }

private:
    MexContext::SharedHandle ctx;
    std::string manufacturer;
    std::string name;
};

}//

#endif//ARRUS_API_MATLAB_WRAPPERS_DEVICES_PROBE_PROBE_MODEL_ID_H
