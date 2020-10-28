#include "arrus/core/api/io/settings.h"
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <cstdlib>

#include "arrus/core/common/logging.h"
#include "arrus/core/session/SessionSettings.h"
#include "arrus/common/utils.h"

#ifdef _MSC_VER

#include <io.h>
#define ARRUS_OPEN_FILE _open

#elif ARRUS_LINUX

#include <fcntl.h>
#define ARRUS_OPEN_FILE open

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

using namespace ::arrus::devices;
using namespace ::arrus::session;

template<typename T>
std::unique_ptr<T> readProtoTxt(const std::string &filepath) {
    int fd = ARRUS_OPEN_FILE(filepath.c_str(), O_RDONLY);
    ARRUS_REQUIRES_TRUE(
        fd != 0, arrus::format("Could not open file {}", filepath));
    google::protobuf::io::FileInputStream input(fd);
    input.SetCloseOnDelete(true);
    auto result = std::make_unique<T>();
    google::protobuf::TextFormat::Parse(&input, result.get());
    return result;
}

ProbeAdapterSettings
readAdapterSettings(const ap::ProbeAdapterModel &proto) {
    ProbeAdapterModelId id(proto.id().manufacturer(), proto.id().name());
    // Safe, should be verified by probe adapter proto validator.
    auto nChannels = static_cast<ChannelIdx>(proto.n_channels());

    ProbeAdapterSettings::ChannelMapping channelMapping;
    using ChannelAddress = ProbeAdapterSettings::ChannelAddress;

    if(proto.has_channel_mapping()) {
        const auto &mapping = proto.channel_mapping();
        const auto &us4oems = mapping.us4oems();
        const auto &inChannels = mapping.channels();

        auto modules = ::arrus::castTo<Ordinal>(
            std::begin(us4oems), std::end(us4oems));
        auto channels = ::arrus::castTo<ChannelIdx>(
            std::begin(inChannels), std::end(inChannels));

        ARRUS_REQUIRES_EQUAL(modules.size(), channels.size(),
                             IllegalArgumentException(
                                 "Us4oems and channels lists should have "
                                 "the same size"));
        channelMapping = std::vector<ChannelAddress>{modules.size()};
        for(int i = 0; i < modules.size(); ++i) {
            channelMapping[i] = {modules[i], channels[i]};
        }
    } else if(!proto.channel_mapping_regions().empty()) {
        std::vector<Ordinal> modules;
        std::vector<ChannelIdx> channels;
        for(auto const &region : proto.channel_mapping_regions()) {
            auto module = static_cast<Ordinal>(region.us4oem());

            if(region.has_region()) {

                ChannelIdx begin = ARRUS_SAFE_CAST(region.region().begin(), ChannelIdx);
                ChannelIdx end = ARRUS_SAFE_CAST(region.region().end(), ChannelIdx);

                for(ChannelIdx ch = begin; ch <= end; ++ch) {
                    channelMapping.emplace_back(module, ch);
                }
            }
            else {
                // Just channels.
                for(auto channel : region.channels()) {
                    channelMapping.emplace_back(
                        module, static_cast<ChannelIdx>(channel));
                }
            }

        }
    }
    return ProbeAdapterSettings(id, nChannels, channelMapping);
}


ProbeModel readProbeModel(const proto::ProbeModel &proto) {
    ProbeModelId id{proto.id().manufacturer(), proto.id().name()};
    using ElementIdxType = ProbeModel::ElementIdxType;

    auto nElementsVec = ::arrus::castTo<ElementIdxType>(
        std::begin(proto.n_elements()), std::end(proto.n_elements()));
    // TODO move
    Tuple<ElementIdxType> nElements{nElementsVec};

    std::vector<double> pitchVec(proto.pitch().size());
    std::copy(std::begin(proto.pitch()), std::end(proto.pitch()),
              std::begin(pitchVec));
    Tuple<double> pitch{pitchVec};

    ::arrus::Interval<float> txFreqRange{static_cast<float>(proto.tx_frequency_range().begin()),
                                         static_cast<float>(proto.tx_frequency_range().end())};
    ::arrus::Interval<uint8> voltageRange{static_cast<uint8>(proto.voltage_range().begin()),
                                          static_cast<uint8>(proto.voltage_range().end())};
    return ProbeModel(id, nElements, pitch, txFreqRange, voltageRange);
}


std::vector<ChannelIdx> readProbeConnectionChannelMapping(
    const ap::ProbeToAdapterConnection &connection) {

    const auto &channelMapping = connection.channel_mapping();
    const auto &ranges = connection.channel_mapping_ranges();

    if(!channelMapping.empty()) {
        return castTo<ChannelIdx>(std::begin(channelMapping),
                                  std::end(channelMapping));
    } else if(!ranges.empty()) {
        std::vector<ChannelIdx> result;
        for(auto const &range: ranges) {
            for(int i = range.begin(); i <= range.end(); ++i) {
                result.push_back(static_cast<ChannelIdx>(i));
            }
        }
        return result;
    } else {
        throw ArrusException("NYI");
    }
}

SettingsDictionary
readDictionary(const ap::Dictionary *proto) {
    SettingsDictionary result;

    if(proto == nullptr) {
        return result;
    }

    for(auto const &adapter : proto->probe_adapter_models()) {
        result.insertAdapterSettings(readAdapterSettings(adapter));
    }

    // index connections
    std::unordered_multimap<std::string, const ap::ProbeToAdapterConnection *> connections;

    for(const ap::ProbeToAdapterConnection &conn : proto->probe_to_adapter_connections()) {
        std::string key =
            SettingsDictionary::convertProtoIdToString(conn.probe_model_id());
        const ap::ProbeToAdapterConnection *ptr = &conn;
        connections.emplace(key, ptr);
    }

    // Read probes.
    for(auto const &probe : proto->probe_models()) {
        const ProbeModel probeModel = readProbeModel(probe);
        result.insertProbeModel(probeModel);

        std::string key =
            SettingsDictionary::convertProtoIdToString(probe.id());
        auto range = connections.equal_range(key);
        for(auto it = range.first; it != range.second; ++it) {
            auto conn = it->second;
            std::vector<ChannelIdx> channelMapping =
                readProbeConnectionChannelMapping(*conn);

            for(auto const &adapterProtoId : conn->probe_adapter_model_id()) {
                const ProbeAdapterModelId adapterId(
                    adapterProtoId.manufacturer(),
                    adapterProtoId.name());
                result.insertProbeSettings(
                    ProbeSettings(probeModel, channelMapping), adapterId);
            }
        }
    }
    return result;
}

RxSettings readRxSettings(const proto::RxSettings &proto) {
    std::optional<uint16> dtgcAtt;
    if(proto.dtgcAttenuation__case() == proto::RxSettings::kDtgcAttenuation) {
        // dtgc attenuation is set
        dtgcAtt = static_cast<uint16>(proto.dtgc_attenuation());
    }
    auto pgaGain = static_cast<uint16>(proto.pga_gain());
    auto lnaGain = static_cast<uint16>(proto.lna_gain());

    RxSettings::TGCCurve tgcSamples = castTo<arrus::ops::us4r::TGCSampleValue>(
        std::begin(proto.tgc_samples()), std::end(proto.tgc_samples()));

    uint32 lpfCutoff = proto.lpf_cutoff();

    std::optional<uint16> activeTermination;
    if(proto.activeTermination__case() ==
       proto::RxSettings::kActiveTermination) {
        activeTermination = static_cast<uint16>(proto.active_termination());
    }

    return RxSettings(dtgcAtt, pgaGain, lnaGain, tgcSamples, lpfCutoff,
                      activeTermination);
}

ProbeAdapterSettings readOrGetAdapterSettings(const proto::Us4RSettings &us4r,
                                              const SettingsDictionary &dictionary) {
    if(us4r.has_adapter()) {
        return readAdapterSettings(us4r.adapter());
    } else if(us4r.has_adapter_id()) {
        ProbeAdapterModelId id{us4r.adapter_id().manufacturer(),
                               us4r.adapter_id().name()};
        try {
            return dictionary.getAdapterSettings(id);
        } catch(const std::out_of_range &) {
            throw IllegalArgumentException(
                arrus::format("Adapter with id {} not found.", id.toString()));
        }
    } else {
        throw ArrusException("NYI");
    }
}

ProbeSettings readOrGetProbeSettings(const proto::Us4RSettings &us4r,
                                     const ProbeAdapterModelId &adapterId,
                                     const SettingsDictionary &dictionary) {
    if(us4r.has_probe()) {
        ProbeModel model = readProbeModel(us4r.probe());
        std::vector<ChannelIdx> channelMapping =
            readProbeConnectionChannelMapping(
                us4r.probe_to_adapter_connection());
        return ProbeSettings(model, channelMapping);
    } else if(us4r.has_probe_id()) {
        ProbeModelId id{us4r.probe_id().manufacturer(), us4r.probe_id().name()};
        if(us4r.has_probe_to_adapter_connection()) {
            std::vector<ChannelIdx> channelMapping =
                readProbeConnectionChannelMapping(
                    us4r.probe_to_adapter_connection());
            try {
                ProbeModel model = dictionary.getProbeModel(id);
                return ProbeSettings(model, channelMapping);
            } catch(std::out_of_range &) {
                throw IllegalArgumentException(
                    arrus::format("Probe with id {} not found.",
                                  id.toString()));
            }
        } else {
            try {
                return dictionary.getProbeSettings(id, adapterId);
            } catch(std::out_of_range &) {
                throw IllegalArgumentException(
                    arrus::format(
                        "Probe settings for probe with id {}"
                        " adapter with id {} not found.", id.toString()));
            }
        }
    } else {
        throw ArrusException("NYI");
    }
}

template<typename T>
std::vector<T> readChannelsMask(const proto::Us4RSettings_ChannelsMask &mask) {
    auto &channels = mask.channels();

    // validate
    for(auto channel : channels) {
        ARRUS_REQUIRES_DATA_TYPE(channel, T, arrus::format(
            "Channel mask should contain only values from uint16 range "
            "(found: '{}')", channel));
    }
    std::vector<T> result;

    for(auto channel : channels) {
        result.push_back(static_cast<T>(channel));
    }
    return result;
}


Us4RSettings readUs4RSettings(const proto::Us4RSettings &us4r,
                              const SettingsDictionary &dictionary) {
    std::optional<HVSettings> hvSettings;
    if(us4r.has_hv()) {
        auto &manufacturer = us4r.hv().model_id().manufacturer();
        auto &name = us4r.hv().model_id().name();
        ARRUS_REQUIRES_NON_EMPTY_IAE(manufacturer);
        ARRUS_REQUIRES_NON_EMPTY_IAE(name);
        hvSettings = HVSettings(HVModelId(manufacturer, name));
    }
    if(!us4r.us4oems().empty()) {
        // Us4OEMs are provided.
        std::vector<Us4OEMSettings> us4oemSettings;

        std::vector<std::unordered_set<uint8>> us4oemChannelsMask;
        us4oemChannelsMask.resize(us4r.us4oems().size());

        if(!us4r.us4oem_channels_mask().empty()
            && us4r.us4oems().size() != us4r.us4oem_channels_mask().size()) {
            throw ::arrus::IllegalArgumentException(
                "The number of us4oem channels "
                "masks should be the same as the number of us4oems.");
        }

        int i = 0;
        for(auto &mask: us4r.us4oem_channels_mask()) {
            auto channelsMask = readChannelsMask<uint8>(mask);

            us4oemChannelsMask[i] = std::unordered_set(
                std::begin(channelsMask), std::end(channelsMask));
            ++i;
        }

        for(auto const &us4oem : us4r.us4oems()) {
            auto rxSettings = readRxSettings(us4oem.rx_settings());
            auto channelMapping = castTo<ChannelIdx>(
                std::begin(us4oem.channel_mapping()),
                std::end(us4oem.channel_mapping()));
            auto activeChannelGroups = castTo<bool>(
                std::begin(us4oem.active_channel_groups()),
                std::end(us4oem.active_channel_groups()));
            us4oemSettings.emplace_back(
                channelMapping, activeChannelGroups, rxSettings,
                us4oemChannelsMask[i]);
        }
        return Us4RSettings(us4oemSettings, hvSettings);
    } else {
        ProbeAdapterSettings adapterSettings =
            readOrGetAdapterSettings(us4r, dictionary);
        ProbeSettings probeSettings = readOrGetProbeSettings(
            us4r, adapterSettings.getModelId(), dictionary);
        RxSettings rxSettings = readRxSettings(us4r.rx_settings());

        // ensure that user provided channels mask
        // TODO(pjarosik) consider removing this check in the future
        if(!us4r.has_channels_mask()) {
            throw IllegalArgumentException(
                "Us4r settings field 'channels_mask' is required. "
                "Set empty array of channels if you want to turn off channel masking.");
        }

        if(us4r.us4oem_channels_mask().empty()) {
            throw IllegalArgumentException(
                "Us4r settings field 'us4oem_channels_mask is required. "
                "Set empty array of channels for each of the module explicitly if you want "
                "to turn of channel masking.");
        }

        std::vector<ChannelIdx> channelsMask =
            readChannelsMask<ChannelIdx>(us4r.channels_mask());
        std::vector<std::vector<uint8>> us4oemChannelsMask;
        for(auto &mask: us4r.us4oem_channels_mask()) {
            us4oemChannelsMask.push_back(readChannelsMask<uint8>(mask));
        }

        return Us4RSettings(adapterSettings, probeSettings, rxSettings,
                            hvSettings, channelsMask, us4oemChannelsMask);
    }
}



SessionSettings readSessionSettings(const std::string &filepath) {
    auto logger = ::arrus::getDefaultLogger();
    // Read and validate session.
    std::filesystem::path sessionSettingsPath{filepath};
    if(!std::filesystem::is_regular_file(sessionSettingsPath)) {
        throw IllegalArgumentException(
            ::arrus::format("File not found {}.", filepath));
    }
    std::unique_ptr<ap::SessionSettings> s =
        readProtoTxt<ap::SessionSettings>(filepath);

    //Validate.
    SessionSettingsProtoValidator validator(
        "session settings in " + filepath);
    validator.validate(s);
    validator.throwOnErrors();

    // Read and validate Dictionary.
    std::unique_ptr<ap::Dictionary> d;
    if(!s->dictionary_file().empty()) {
        std::string dictionaryPathStr;
        // 1. Try to use the parent directory of session settings.
        auto dictP = sessionSettingsPath.parent_path() / s->dictionary_file();
        if(std::filesystem::is_regular_file(dictP)) {
            dictionaryPathStr = dictP.u8string();
        } else {
            // 2. Try to use ARRUS_PATH, if available.
            const char *arrusP = std::getenv(ARRUS_PATH_KEY);
            if(arrusP != nullptr) {
                std::filesystem::path arrusDicP{arrusP};
                arrusDicP = arrusDicP / s->dictionary_file();
                if(std::filesystem::is_regular_file(arrusDicP)) {
                    dictionaryPathStr = arrusDicP.u8string();
                } else {
                    throw IllegalArgumentException(
                        ::arrus::format("Invalid path to dictionary: {}",
                                        s->dictionary_file()));
                }
            } else {
                throw IllegalArgumentException(
                    ::arrus::format("Invalid path to dictionary: {}",
                                    s->dictionary_file()));
            }
        }
        d = readProtoTxt<ap::Dictionary>(dictionaryPathStr);
        DictionaryProtoValidator dictionaryValidator("dictionary");
        dictionaryValidator.validate(d);
        dictionaryValidator.throwOnErrors();
    }

    SettingsDictionary dictionary = readDictionary(d.get());

    Us4RSettings us4rSettings = readUs4RSettings(s->us4r(), dictionary);
    // TODO std move

    SessionSettings sessionSettings(us4rSettings);

    logger->log(LogSeverity::DEBUG,
                arrus::format("Read settings from '{}': {}",
                              filepath, ::arrus::toString(sessionSettings)));

    return sessionSettings;
}


}