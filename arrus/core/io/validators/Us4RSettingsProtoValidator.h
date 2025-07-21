#ifndef ARRUS_CORE_IO_VALIDATORS_US4RSETTINGSPROTOVALIDATOR_H
#define ARRUS_CORE_IO_VALIDATORS_US4RSETTINGSPROTOVALIDATOR_H

#include "arrus/common/compiler.h"
#include "arrus/core/common/validation.h"

#include "arrus/core/io/validators/ProbeModelProtoValidator.h"
#include "arrus/core/io/validators/ProbeAdapterModelProtoValidator.h"
#include "arrus/core/io/validators/ProbeToAdapterConnectionProtoValidator.h"
#include "arrus/core/io/validators/RxSettingsProtoValidator.h"
#include "arrus/core/io/validators/GpuSettingsProtoValidator.h"


COMPILER_PUSH_DIAGNOSTIC_STATE
COMPILER_DISABLE_MSVC_WARNINGS(4127)
#include "io/proto/devices/us4r/Us4RSettings.pb.h"
COMPILER_POP_DIAGNOSTIC_STATE

namespace arrus::io {

class Us4RSettingsProtoValidator : public Validator<arrus::proto::Us4RSettings> {
 public:
    explicit Us4RSettingsProtoValidator(const std::string &name) : Validator(name) {}

    void validate(const arrus::proto::Us4RSettings &us4r) override {
        bool hasProbeSettings = hasProbe(us4r) || hasProbeId(us4r);
        bool hasAdapterSettings = us4r.has_adapter() || us4r.has_adapter_id();
        if (hasErrors()) {
            return;
        }
        bool hasAllProbeSettings = hasProbeSettings && hasAdapterSettings && us4r.has_rx_settings();
        expectTrue("us4r", hasAllProbeSettings,
                   "All of the following fields are required: "
                   "(probe settings, adapter settings, rx settings)");

        if (hasProbe(us4r) || us4r.has_adapter()) {
            // Custom probe or adapter
            expectTrue("us4r", hasProbeToAdapterConnection(us4r),
                       "Probe to adapter connection is required "
                       "for custom probe and adapter definitions.");
        }

        if (hasErrors()) {
            return;
        }

        for (auto &probe: us4r.probe()) {
            ProbeModelProtoValidator probeValidator("custom probe");
            probeValidator.validate(probe);
            copyErrorsFrom(probeValidator);
        }
        if (us4r.has_adapter()) {
            expectTrue("adapter_id", !us4r.has_adapter_id(),
                       "Adapter Id should not be set "
                       "(custom adapter already set).");

            ProbeAdapterModelProtoValidator adapterValidator("custom adapter");
            auto &adapter = us4r.adapter();
            adapterValidator.validate(adapter);
            copyErrorsFrom(adapterValidator);
        } else {
            expectTrue("adapter_id", us4r.has_adapter_id(), "Adapter id or custom adapter def. is required.");
        }

        // otherwise it should overwrite the settings from dictionary.
        for (auto &conn: us4r.probe_to_adapter_connection()) {
            // probe and adapter ids are forbidden (to avoid any additional confusion).
            ProbeToAdapterConnectionProtoValidator connValidator("custom connection");
            connValidator.validate(conn);
            copyErrorsFrom(connValidator);
        }

        expectTrue("rx_settings", us4r.has_rx_settings(), "Rx settings are required.");
        RxSettingsProtoValidator rxSettingsValidator("rx_settings");
        rxSettingsValidator.validate(us4r.rx_settings());
        copyErrorsFrom(rxSettingsValidator);

        // Validate GPU settings if present
        GpuSettingsProtoValidator gpuSettingsValidator("gpu");
        gpuSettingsValidator.validate(us4r.gpu());
        copyErrorsFrom(gpuSettingsValidator);        
    }

    bool hasProbe(const arrus::proto::Us4RSettings &us4r) const {
        return us4r.probe_size() > 0;
    }

    bool hasProbeId(const arrus::proto::Us4RSettings &us4r) const {
        return us4r.probe_id_size() > 0;
    }

    bool hasProbeToAdapterConnection(const arrus::proto::Us4RSettings &us4r) const {
        return us4r.probe_to_adapter_connection_size() > 0;
    }
};

}

#endif//ARRUS_CORE_IO_VALIDATORS_US4RSETTINGSPROTOVALIDATOR_H
