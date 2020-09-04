#ifndef ARRUS_CORE_IO_VALIDATORS_SESSIONSETTINGSPROTOVALIDATOR_H
#define ARRUS_CORE_IO_VALIDATORS_SESSIONSETTINGSPROTOVALIDATOR_H

#include "arrus/common/compiler.h"
#include "arrus/core/common/validation.h"

#include "arrus/core/io/validators/ProbeModelProtoValidator.h"
#include "arrus/core/io/validators/ProbeAdapterModelProtoValidator.h"
#include "arrus/core/io/validators/ProbeToAdapterConnectionProtoValidator.h"
#include "arrus/core/io/validators/RxSettingsProtoValidator.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
COMPILER_DISABLE_MSVC_WARNINGS(4127)

#include "io/proto/session/SessionSettings.pb.h"

COMPILER_POP_DIAGNOSTIC_STATE


namespace arrus::io {

class SessionSettingsProtoValidator
    : public Validator<std::unique_ptr<arrus::proto::SessionSettings>> {

    public:
    explicit SessionSettingsProtoValidator(const std::string &name)
        : Validator(name) {}

    void validate(
        const std::unique_ptr<arrus::proto::SessionSettings> &obj) override {
        expectTrue("us4r", obj->has_us4r(), "us4r settings are required");

        // TODO extract us4r settings validator

        if(hasErrors()) {
            return;
        }

        auto &us4r = obj->us4r();

        bool hasUs4oemSettings = !us4r.us4oems().empty();
        bool hasProbeSettings = us4r.has_probe() || us4r.has_probe_id();
        bool hasAdapterSettings = us4r.has_adapter() || us4r.has_adapter_id();
        bool hasProbeAdapterSettings = hasProbeSettings || hasAdapterSettings ||
                                       us4r.has_probe_to_adapter_connection() ||
                                       us4r.has_rx_settings();


        expectTrue("us4r", !(hasUs4oemSettings ^ hasProbeAdapterSettings),
                   "Exactly one of the following should be set in us4r "
                   "settings: a list of us4oem settings or: (probe settings, "
                   "adapter settings, probe<->adapter connection, rx settings)"
        );

        if(hasErrors()) {
            return;
        }

        if(hasUs4oemSettings) {
            int i = 0;
            for(auto &settings : us4r.us4oems()) {
                std::string fieldName = "Us4OEM:" + std::to_string(i);
                expectTrue(fieldName, settings.has_rx_settings(),
                           "Rx settings are required.");

                expectAllDataType<ChannelIdx>(fieldName,
                    std::begin(settings.channel_mapping()),
                    std::end(settings.channel_mapping()),
                    "channel_mapping");

                RxSettingsProtoValidator validator(fieldName);
                validator.validate(settings.rx_settings());
                copyErrorsFrom(validator);
                ++i;
            }
        } else if(hasProbeAdapterSettings) {
            bool hasAllProbeSettings = hasProbeSettings && hasAdapterSettings &&
                                       us4r.has_probe_to_adapter_connection() &&
                                       us4r.has_rx_settings();

            expectTrue("us4r", hasAllProbeSettings,
                       "All of the following fields are required: "
                       "(probe settings, adapter settings, "
                       "probe <-> adapter connection, rx settings)");

            if(hasErrors()) {
                return;
            }

            if(us4r.has_probe()) {
                expectTrue("probe_id", !us4r.has_probe_id(),
                           "Probe Id should not be set "
                           "(custom probe already set).");
                ProbeModelProtoValidator probeValidator("custom probe");
                auto &probe = us4r.probe();
                probeValidator.validate(probe);
                copyErrorsFrom(probeValidator);
            } else {
                expectTrue("probe_id", us4r.has_probe_id(),
                           "Probe id or custom probe def. is required.");
            }

            if(us4r.has_adapter()) {
                expectTrue("adapter_id", !us4r.has_adapter_id(),
                           "Adapter Id should not be set "
                           "(custom adapter already set).");

                ProbeAdapterModelProtoValidator adapterValidator(
                    "custom adapter");
                auto &adapter = us4r.adapter();
                adapterValidator.validate(adapter);
                copyErrorsFrom(adapterValidator);
            } else {
                expectTrue("adapter_id", us4r.has_adapter_id(),
                           "Adapter id or custom adapter def. is required.");
            }

            if(us4r.has_probe() && us4r.has_adapter()) {
                expectTrue("probe_to_adapter_connection",
                           us4r.has_probe_to_adapter_connection(),
                           "Custom probe and adapter are set, "
                           "custom probe to adapter connection is required.");
            }

            // otherwise it should overwrite the settings from dictionary.

            if(us4r.has_probe_to_adapter_connection()) {
                // probe and adapter ids are forbidden (to avoid any additional
                // confusion).
                auto &conn = us4r.probe_to_adapter_connection();
                expectTrue("probe_to_adapter_connection",
                           !conn.has_probe_model_id(),
                           "Probe model id is forbidden for custom "
                           "probe connections.");

                expectTrue("probe_to_adapter_connection",
                           conn.probe_adapter_model_id().empty(),
                           "Adapter id is forbidden for custom "
                           "probe connections.");
                ProbeToAdapterConnectionProtoValidator connValidator(
                    "custom connection");
                connValidator.validate(conn);
                copyErrorsFrom(connValidator);
            }

            expectTrue("rx_settings", us4r.has_rx_settings(),
                       "Rx settings are required.");
            RxSettingsProtoValidator rxSettingsValidator("rx_settings");
            rxSettingsValidator.validate(us4r.rx_settings());
            copyErrorsFrom(rxSettingsValidator);
        }
    }
};

}

#endif //ARRUS_CORE_IO_VALIDATORS_SESSIONSETTINGSPROTOVALIDATOR_H
