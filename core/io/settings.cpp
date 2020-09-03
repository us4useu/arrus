#include "arrus/core/api/io/settings.h"
#include <fcntl.h>
#include <memory>

#ifdef _MSC_VER

#include <io.h>

#endif

#include "arrus/common/asserts.h"
#include "arrus/common/format.h"
#include "arrus/common/compiler.h"
#include "arrus/core/common/validation.h"
#include "arrus/core/io/validators/SessionSettingsProtoValidator.h"
#include "arrus/core/io/validators/DictionaryProtoValidator.h"
#include "arrus/core/io/SettingsDictionary.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
COMPILER_DISABLE_MSVC_WARNINGS(4127)

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
// TODO(146) should point to arrus/core/io/...
#include "io/proto/session/SessionSettings.pb.h"
#include "io/proto/Dictionary.pb.h"

COMPILER_POP_DIAGNOSTIC_STATE


namespace arrus::io {

namespace ap = arrus::proto;


template<typename T>
std::unique_ptr<T> readProtoTxt(const std::string &filepath) {
    int fd = open(filepath.c_str(), O_RDONLY);
    ARRUS_REQUIRES_TRUE(
        fd != 0, arrus::format("Could not open file {}", filepath));
    google::protobuf::io::FileInputStream input(fd);
    input.SetCloseOnDelete(true);
    auto result = std::make_unique<T>();
    google::protobuf::TextFormat::Parse(&input, result.get());
    return result;
}

ProbeAdapterSettings readAdapterSettings(const ap::ProbeAdapterModel adapter) {

}

ProbeSettings readProbeSettings(const ap::ProbeModel &probe,
                                const ap::ProbeToAdapterConnection &conn) {

}


SettingsDictionary readDictionary(const std::unique_ptr<ap::Dictionary> proto) {
    SettingsDictionary result;

    if(proto == nullptr) {
        return result;
    }

    for(auto &adapter : proto->probe_adapter_models()) {
        result.insertAdapterSettings(readAdapterSettings(adapter));
    }

    // index connections
    std::unordered_map<std::string, const ap::ProbeToAdapterConnection *> connections;

    for(const ap::ProbeToAdapterConnection &conn : proto->probe_to_adapter_connections()) {
        std::string key =
            SettingsDictionary::convertIdToString(conn.probe_model_id());
        const ap::ProbeToAdapterConnection *ptr = &conn;
        connections.emplace(key, ptr);
    }

    // Read probes.
    for(auto const &probe : proto->probe_models()) {
        const ProbeSettings settings = readProbeSettings()
    }
}

SessionSettings readSessionSettings(const std::string &filepath) {
    // Read and validate session.
    std::unique_ptr<ap::SessionSettings> s =
        readProtoTxt<ap::SessionSettings>(filepath);
    SessionSettingsProtoValidator validator("session settings in " + filepath);
    validator.validate(s);
    validator.throwOnErrors();

    // Read and validate Dictionary.
    std::unique_ptr<ap::Dictionary> d;
    if(!s->dictionary_file().empty()) {
        d = readProtoTxt<ap::Dictionary>(s->dictionary_file());
        DictionaryProtoValidator dictionaryValidator("dictionary");
        dictionaryValidator.validate(d);
        dictionaryValidator.throwOnErrors();
    }

    // read dictionary of settings
    // read us4r settings

    auto &us4r = s->us4r();
    if(!us4r.us4oems().empty()) {
        // Read settings for particular Us4OEMs
        std::vector<Us4OEMSettings> settings
        auto &us4oems = us4r.us4oems();
        for(auto &us4oem: us4oems) {
        }
    } else {
        // Read settings for given probe, adapter and rx settings.
    }

    // if has us4oemsettings -> use us4oemsettings
    // otherwise use probe, adapter, etc.
    // - If probe is provided - use it
    // - otherwise - use probeModelId, get probe model from dictionary file
    // - if not found - raise error, that probe model with this id was not found
    // the same appl

    // if us4oemsettings
//


}


}