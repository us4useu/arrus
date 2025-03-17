#include "arrus/core/api/io/settings.h"
#include <boost/filesystem.hpp>
#include <cstdlib>
#include <fcntl.h>
#include <memory>
#include <unordered_map>

#include "arrus/common/utils.h"
#include "arrus/core/common/logging.h"
#include "arrus/core/session/SessionSettings.h"
#include "cfg/default.h"

#ifdef _MSC_VER

#include <io.h>
#define ARRUS_OPEN_FILE _open

#elif ARRUS_LINUX

#include <fcntl.h>
#define ARRUS_OPEN_FILE open

#endif

#include "arrus/common/asserts.h"
#include "arrus/common/compiler.h"
#include "arrus/common/format.h"
#include "arrus/core/common/validation.h"
#include "arrus/core/io/SettingsDictionary.h"
#include "arrus/core/io/validators/DictionaryProtoValidator.h"
#include "arrus/core/io/validators/SessionSettingsProtoValidator.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
COMPILER_DISABLE_MSVC_WARNINGS(4127)

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
// TODO(146) should point to arrus/core/io/...
#include "io/proto/Dictionary.pb.h"
#include "io/proto/session/SessionSettings.pb.h"

COMPILER_POP_DIAGNOSTIC_STATE

namespace arrus::io {

namespace ap = arrus::proto;

using namespace ::arrus::devices;
using namespace ::arrus::session;

template<typename T>
std::unique_ptr<T> readProtoTxt(const std::string &filepath) {
    int fd = ARRUS_OPEN_FILE(filepath.c_str(), O_RDONLY);
    ARRUS_REQUIRES_TRUE(fd != 0, arrus::format("Could not open file {}", filepath));
    google::protobuf::io::FileInputStream input(fd);
    input.SetCloseOnDelete(true);
    auto result = std::make_unique<T>();
    bool parseOk = google::protobuf::TextFormat::Parse(&input, result.get());
    if (!parseOk) {
        throw IllegalArgumentException(::arrus::format("Error while parsing file {}, please check error messages "
                                                       "that appeared the above.",
                                                       filepath));
    }
    return result;
}

template<typename T>
std::unique_ptr<T> readProtoTxtStr(const std::string &proto) {
    auto result = std::make_unique<T>();
    bool parseOk = google::protobuf::TextFormat::ParseFromString(proto, result.get());
    if (!parseOk) {
        throw IllegalArgumentException("Error while reading proto txt.");
    }
    return result;
}

ProbeAdapterSettings readAdapterSettings(const ap::ProbeAdapterModel &proto) {
    ProbeAdapterModelId id(proto.id().manufacturer(), proto.id().name());
    // Safe, should be verified by probe adapter proto validator.
    auto nChannels = static_cast<ChannelIdx>(proto.n_channels());

    ProbeAdapterSettings::ChannelMapping channelMapping;
    using ChannelAddress = ProbeAdapterSettings::ChannelAddress;

    if (proto.has_channel_mapping()) {
        const auto &mapping = proto.channel_mapping();
        const auto &us4oems = mapping.us4oems();
        const auto &inChannels = mapping.channels();

        auto modules = ::arrus::castTo<Ordinal>(std::begin(us4oems), std::end(us4oems));
        auto channels = ::arrus::castTo<ChannelIdx>(std::begin(inChannels), std::end(inChannels));

        ARRUS_REQUIRES_EQUAL(modules.size(), channels.size(),
                             IllegalArgumentException("Us4oems and channels lists should have "
                                                      "the same size"));
        channelMapping = std::vector < ChannelAddress > {modules.size()};
        for (unsigned i = 0; i < modules.size(); ++i) {
            channelMapping[i] = {modules[i], channels[i]};
        }
    } else if (!proto.channel_mapping_regions().empty()) {
        std::vector<Ordinal> modules;
        std::vector<ChannelIdx> channels;
        for (auto const &region: proto.channel_mapping_regions()) {
            auto module = static_cast<Ordinal>(region.us4oem());

            if (region.has_region()) {

                ChannelIdx begin = ARRUS_SAFE_CAST(region.region().begin(), ChannelIdx);
                ChannelIdx end = ARRUS_SAFE_CAST(region.region().end(), ChannelIdx);

                for (ChannelIdx ch = begin; ch <= end; ++ch) {
                    channelMapping.emplace_back(module, ch);
                }
            } else {
                // Just channels.
                for (auto channel: region.channels()) {
                    channelMapping.emplace_back(module, static_cast<ChannelIdx>(channel));
                }
            }
        }
    }
    // io capabilities
    if (proto.has_io_settings()) {
        auto &ioSettingsProto = proto.io_settings();
        arrus::devices::us4r::IOSettingsBuilder settingsBuilder;
        for (auto &entry: ioSettingsProto.capabilities()) {
            // Convert to arrus IO address.
            std::vector<arrus::devices::us4r::IOAddress> addresses;
            for (auto addressProto: entry.addresses()) {
                Ordinal us4oem = ARRUS_SAFE_CAST(addressProto.us4oem(), Ordinal);
                uint8_t io = ARRUS_SAFE_CAST(addressProto.io(), uint8_t);
                addresses.emplace_back(us4oem, io);
            }
            arrus::devices::us4r::IOAddressSet addressSet(addresses);
            if (addressSet.size() > 0) {
                switch (entry.capability()) {
                    case arrus::proto::IOCapability::PROBE_CONNECTED_CHECK:
                        settingsBuilder.setProbeConnectedCheckCapability(addressSet);
                        break;
                    case arrus::proto::IOCapability::FRAME_METADATA:
                        settingsBuilder.setFrameMetadataCapability(addressSet);
                        break;
                    default:
                        throw IllegalArgumentException(
                            "Unhandled capability nr: " + std::to_string(entry.capability()));
                }
            }
        }
        return ProbeAdapterSettings(id, nChannels, channelMapping, settingsBuilder.build());
    } else {
        return ProbeAdapterSettings(id, nChannels, channelMapping);
    }
}

ProbeModel readProbeModel(const proto::ProbeModel &proto) {
    ProbeModelId id{proto.id().manufacturer(), proto.id().name()};
    using ElementIdxType = ProbeModel::ElementIdxType;

    auto nElementsVec = ::arrus::castTo<ElementIdxType>(std::begin(proto.n_elements()), std::end(proto.n_elements()));
    // TODO move
    Tuple<ElementIdxType> nElements{nElementsVec};

    std::vector<double> pitchVec(proto.pitch().size());
    std::copy(std::begin(proto.pitch()), std::end(proto.pitch()), std::begin(pitchVec));
    Tuple<double> pitch{pitchVec};

    double curvatureRadius = proto.curvature_radius();

    ::arrus::Interval<float> txFreqRange{static_cast<float>(proto.tx_frequency_range().begin()),
                                         static_cast<float>(proto.tx_frequency_range().end())};
    ::arrus::Interval<uint8> voltageRange{static_cast<uint8>(proto.voltage_range().begin()),
                                          static_cast<uint8>(proto.voltage_range().end())};
    return ProbeModel(id, nElements, pitch, txFreqRange, voltageRange, curvatureRadius);
}

std::vector<ChannelIdx> readProbeConnectionChannelMapping(const ap::ProbeToAdapterConnection &connection) {

    const auto &channelMapping = connection.channel_mapping();
    const auto &ranges = connection.channel_mapping_ranges();

    if (!channelMapping.empty()) {
        return castTo<ChannelIdx>(std::begin(channelMapping), std::end(channelMapping));
    } else if (!ranges.empty()) {
        std::vector<ChannelIdx> result;
        for (auto const &range: ranges) {
            for (int i = range.begin(); i <= range.end(); ++i) {
                result.push_back(static_cast<ChannelIdx>(i));
            }
        }
        return result;
    } else {
        throw ArrusException("NYI");
    }
}

std::unordered_multimap<std::string, ap::ProbeToAdapterConnection> indexProbeToAdapterConnections(
    const google::protobuf::RepeatedPtrField<::arrus::proto::ProbeToAdapterConnection> &probeToAdapterConnections) {
    std::unordered_multimap<std::string, ap::ProbeToAdapterConnection> connections;
    for (const ap::ProbeToAdapterConnection &conn: probeToAdapterConnections) {
        std::string key = SettingsDictionary::convertProtoIdToString(conn.probe_model_id());
        connections.emplace(key, conn);
    }
    return connections;
}

SettingsDictionary readDictionary(const ap::Dictionary *proto) {
    SettingsDictionary result;

    if (proto == nullptr) {
        return result;
    }

    for (auto const &adapter: proto->probe_adapter_models()) {
        result.insertAdapterSettings(readAdapterSettings(adapter));
    }

    // index connections
    auto &probeToAdapterConnections = proto->probe_to_adapter_connections();

    std::unordered_multimap<std::string, ap::ProbeToAdapterConnection> connections =
        indexProbeToAdapterConnections(probeToAdapterConnections);

    // Read probes.
    for (auto const &probe: proto->probe_models()) {
        const ProbeModel probeModel = readProbeModel(probe);
        result.insertProbeModel(probeModel);
        std::string key = SettingsDictionary::convertProtoIdToString(probe.id());
        auto range = connections.equal_range(key);
        for (auto it = range.first; it != range.second; ++it) {
            auto conn = it->second;
            std::vector<ChannelIdx> channelMapping = readProbeConnectionChannelMapping(conn);

            for (auto const &adapterProtoId: conn.probe_adapter_model_id()) {
                const ProbeAdapterModelId adapterId(adapterProtoId.manufacturer(), adapterProtoId.name());
                result.insertProbeSettings(ProbeSettings(probeModel, channelMapping), adapterId);
            }
        }
    }
    return result;
}

RxSettings readRxSettings(const proto::RxSettings &proto) {
    std::optional<uint16> dtgcAtt;
    if (proto.dtgcAttenuation__case() == proto::RxSettings::kDtgcAttenuation) {
        // dtgc attenuation is set
        dtgcAtt = static_cast<uint16>(proto.dtgc_attenuation());
    }
    auto pgaGain = static_cast<uint16>(proto.pga_gain());
    auto lnaGain = static_cast<uint16>(proto.lna_gain());

    RxSettings::TGCCurve tgcSamples =
        castTo<arrus::ops::us4r::TGCSampleValue>(std::begin(proto.tgc_samples()), std::end(proto.tgc_samples()));

    uint32 lpfCutoff = proto.lpf_cutoff();

    std::optional<uint16> activeTermination;
    if (proto.activeTermination__case() == proto::RxSettings::kActiveTermination) {
        activeTermination = static_cast<uint16>(proto.active_termination());
    }
    // TODO apply characteristic parameter
    return RxSettings(dtgcAtt, pgaGain, lnaGain, tgcSamples, lpfCutoff, activeTermination);
}

ProbeAdapterSettings readOrGetAdapterSettings(const proto::Us4RSettings &us4r, const SettingsDictionary &dictionary) {
    if (us4r.has_adapter()) {
        return readAdapterSettings(us4r.adapter());
    } else if (us4r.has_adapter_id()) {
        ProbeAdapterModelId id{us4r.adapter_id().manufacturer(), us4r.adapter_id().name()};
        try {
            return dictionary.getAdapterSettings(id);
        } catch (const std::out_of_range &) {
            throw IllegalArgumentException(arrus::format("Adapter with id {} not found.", id.toString()));
        }
    } else {
        throw ArrusException("NYI");
    }
}

proto::ProbeToAdapterConnection
getUniqueConnection(const proto::ProbeModel_Id &probeId,
                    std::unordered_multimap<std::string, ap::ProbeToAdapterConnection> &connections) {
    std::string key = SettingsDictionary::convertProtoIdToString(probeId);
    ProbeModelId id{probeId.manufacturer(), probeId.name()};
    if (connections.count(key) > 1) {
        throw IllegalArgumentException(
            format("Multiple probe to adapter connection definitions for probe: {}", id.toString()));
    }
    auto it = connections.find(key);
    if (it == std::end(connections)) {
        throw IllegalArgumentException(
            format("No definition found for probe {}, but the probe is used in the probe to adapter connections list. ",
                   id.toString()));
    }
    return it->second;
}

std::vector<ProbeSettings> readOrGetProbeSettings(const proto::Us4RSettings &us4r, const ProbeAdapterModelId &adapterId,
                                                  const SettingsDictionary &dictionary) {
    // Read and index by probe to adapter connections
    std::unordered_multimap<std::string, ap::ProbeToAdapterConnection> connections =
        indexProbeToAdapterConnections(us4r.probe_to_adapter_connection());

    std::vector<ProbeSettings> result;

    int nProbes = us4r.probe_id_size() + us4r.probe_size();
    ARRUS_REQUIRES_TRUE(us4r.probe_to_adapter_connection_size() == 0
                            || (us4r.probe_to_adapter_connection_size() == nProbes),
                        "You should skip probe to adapter connections list, or provide the same number of "
                        "probe to adapter connections and probe definitions in the configuration file.");

    for (auto &probeId: us4r.probe_id()) {
        ProbeModelId id{probeId.manufacturer(), probeId.name()};
        ProbeModel model = dictionary.getProbeModel(id);
        if (us4r.probe_to_adapter_connection_size() > 0) {
            // If there are some probe to adapter connections: use only them.
            auto connection = getUniqueConnection(probeId, connections);
            std::vector<ChannelIdx> channelMapping = readProbeConnectionChannelMapping(connection);
            std::optional<BitstreamId> bitstreamId;
            if (connection.has_bitstream_id()) {
                bitstreamId = ARRUS_SAFE_CAST(connection.bitstream_id().ordinal(), uint16_t);
            }
            result.emplace_back(model, channelMapping, bitstreamId);
        } else {
            // Otherwise, for probe to adapter connection use dictionary only.
            result.push_back(dictionary.getProbeSettings(id, adapterId));
        }
    }

    for (auto &probe: us4r.probe()) {
        ProbeModel model = readProbeModel(probe);
        const auto &probeId = probe.id();
        proto::ProbeToAdapterConnection connection;
        if (us4r.probe_to_adapter_connection_size() == 1) {
            // Also a single probe (see the verification above).
            connection = *std::begin(us4r.probe_to_adapter_connection());
        } else {
            connection = getUniqueConnection(probeId, connections);
        }
        std::vector<ChannelIdx> channelMapping = readProbeConnectionChannelMapping(connection);
        std::optional<BitstreamId> bitstreamId;
        if (connection.has_bitstream_id()) {
            bitstreamId = ARRUS_SAFE_CAST(connection.bitstream_id().ordinal(), uint16_t);
        }
        result.emplace_back(model, channelMapping, bitstreamId);
    }
    return result;
}

template<typename T>
std::vector<std::unordered_set<T>> readChannelsMask(const proto::Us4RSettings &us4r) {
    std::vector<std::unordered_set<T>> result;
    for (const auto &mask: us4r.channels_mask()) {
        auto &channels = mask.channels();
        // validate
        for (auto channel: channels) {
            ARRUS_REQUIRES_DATA_TYPE(channel, T,
                                     arrus::format("Channel mask should contain only values from uint16 range "
                                                   "(found: '{}')",
                                                   channel));
        }
        std::unordered_set<T> probeMask;
        for (auto channel: channels) {
            probeMask.insert(static_cast<T>(channel));
        }
        result.emplace_back(std::move(probeMask));
    }
    return result;
}

Us4OEMSettings::ReprogrammingMode convertToReprogrammingMode(proto::Us4OEMSettings_ReprogrammingMode mode);

ProbeModel readProbeModel(const proto::FileSettings &file, const SettingsDictionary &dictionary) {
    if (file.has_probe()) {
        return readProbeModel(file.probe());
    } else if (file.has_probe_id()) {
        ProbeModelId id{file.probe_id().manufacturer(), file.probe_id().name()};
        try {
            return dictionary.getProbeModel(id);
        } catch (std::out_of_range &) {
            throw IllegalArgumentException(format("Probe model with id {} not found.", id.toString()));
        }
    } else {
        throw std::runtime_error("NYI");
    }
}

FileSettings readFileSettings(const proto::FileSettings &file, const SettingsDictionary &dictionary) {
    return FileSettings{file.filepath(), file.n_frames(), readProbeModel(file, dictionary)};
}

std::vector<Bitstream>
readBitstreams(const ::google::protobuf::RepeatedPtrField<::arrus::proto::Bitstream> &bitstreams) {
    std::vector<Bitstream> result;
    for (auto &b: bitstreams) {
        std::vector<uint8> levels;
        std::vector<uint16> periods;
        std::transform(std::begin(b.levels()), std::end(b.levels()), std::back_inserter(levels),
                       [](auto v) { return (uint8)v; });
        std::transform(std::begin(b.periods()), std::end(b.periods()), std::back_inserter(periods),
                       [](auto v) { return (uint16)v; });
        result.emplace_back(levels, periods);
    }
    return result;
}

Interval<float> getInterval(const ::arrus::proto::IntervalDouble& interval) {
    return Interval<float>{static_cast<float>(interval.begin()), static_cast<float>(interval.end())};
}

template<typename T>
Interval<T> getInterval(const ::arrus::proto::IntervalInteger& interval) {
    return Interval<T>{static_cast<T>(interval.begin()), static_cast<T>(interval.end())};
}

std::optional<Us4RTxRxLimits>
readUs4RTxRxLimits(const proto::Us4RSettings &us4r) {
    if(us4r.has_tx_rx_limits()) {
        auto &limits = us4r.tx_rx_limits();
        std::optional<Interval<float>> pri, pulseLength;
        std::optional<Interval<Voltage>> voltage;
        if(limits.has_pri()) {
            pri = getInterval(limits.pri());
        }
        if(limits.has_pulse_length()) {
            pulseLength = getInterval(limits.pulse_length());
        }
        if(limits.has_voltage()) {
            voltage = getInterval<Voltage>(limits.voltage());
        }
        return Us4RTxRxLimits{pulseLength, voltage, pri};
    } else {
        return std::nullopt;
    }
}

Us4RSettings readUs4RSettings(const proto::Us4RSettings &us4r, const SettingsDictionary &dictionary) {
    std::optional<HVSettings> hvSettings;
    std::optional<DigitalBackplaneSettings> digitalBackplaneSettings;
    std::optional<Ordinal> nUs4OEMs;
    std::vector<Ordinal> adapterToUs4RModuleNr;
    int txFrequencyRange = 1;

    if (us4r.has_hv()) {
        auto &manufacturer = us4r.hv().model_id().manufacturer();
        auto &name = us4r.hv().model_id().name();
        ARRUS_REQUIRES_NON_EMPTY_IAE(manufacturer);
        ARRUS_REQUIRES_NON_EMPTY_IAE(name);
        hvSettings = HVSettings(HVModelId(manufacturer, name));
    }
    if (us4r.has_digital_backplane()) {
        auto &manufacturer = us4r.digital_backplane().model_id().manufacturer();
        auto &name = us4r.digital_backplane().model_id().name();
        ARRUS_REQUIRES_NON_EMPTY_IAE(manufacturer);
        ARRUS_REQUIRES_NON_EMPTY_IAE(name);
        digitalBackplaneSettings = DigitalBackplaneSettings(DigitalBackplaneId(manufacturer, name));
    }
    if (us4r.optional_nus4ems_case() != proto::Us4RSettings::OPTIONAL_NUS4EMS_NOT_SET) {
        nUs4OEMs = static_cast<Ordinal>(us4r.nus4oems());
    }
    if (us4r.optional_tx_frequency_range_case() != proto::Us4RSettings::OPTIONAL_TX_FREQUENCY_RANGE_NOT_SET) {
        txFrequencyRange = static_cast<int>(us4r.tx_frequency_range());
    }
    if (!us4r.adapter_to_us4r_module_nr().empty()) {
        auto &adapter2Us4RModule = us4r.adapter_to_us4r_module_nr();
        for (auto &nr: adapter2Us4RModule) {
            adapterToUs4RModuleNr.emplace_back(static_cast<Ordinal>(nr));
        }
    }
    WatchdogSettings watchdog = WatchdogSettings::defaultSettings();
    if(us4r.has_watchdog()) {
        auto enabled = us4r.watchdog().enabled();
        if(enabled) {
            watchdog = WatchdogSettings{
                ARRUS_SAFE_CAST(us4r.watchdog().oem_threshold0(), float),
                ARRUS_SAFE_CAST(us4r.watchdog().oem_threshold1(), float),
                ARRUS_SAFE_CAST(us4r.watchdog().host_threshold(), float)
            };
        }
        else {
            watchdog = WatchdogSettings::disabled();
        }
    }

    ProbeAdapterSettings adapterSettings = readOrGetAdapterSettings(us4r, dictionary);
    std::vector<ProbeSettings> probeSettings =
        readOrGetProbeSettings(us4r, adapterSettings.getModelId(), dictionary);
    RxSettings rxSettings = readRxSettings(us4r.rx_settings());

    std::vector<std::unordered_set<ChannelIdx>> channelsMask = readChannelsMask<ChannelIdx>(us4r);
    std::vector<std::vector<uint8>> us4oemChannelsMask;
    auto reprogrammingMode = convertToReprogrammingMode(us4r.reprogramming_mode());
    std::vector<Bitstream> bitstreams = readBitstreams(us4r.bitstreams());
    std::optional<Us4RTxRxLimits> limits = readUs4RTxRxLimits(us4r);

    return {adapterSettings,
            probeSettings,
            rxSettings,
            hvSettings,
            channelsMask,
            reprogrammingMode,
            nUs4OEMs,
            adapterToUs4RModuleNr,
            us4r.external_trigger(),
            txFrequencyRange,
            digitalBackplaneSettings,
            bitstreams,
            limits,
            watchdog
    };
}
Us4OEMSettings::ReprogrammingMode convertToReprogrammingMode(proto::Us4OEMSettings_ReprogrammingMode mode) {
    switch (mode) {
        case proto::Us4OEMSettings_ReprogrammingMode_SEQUENTIAL: return Us4OEMSettings::ReprogrammingMode::SEQUENTIAL;
        case proto::Us4OEMSettings_ReprogrammingMode_PARALLEL: return Us4OEMSettings::ReprogrammingMode::PARALLEL;
        default: throw std::runtime_error("Unknown reprogramming mode: " + std::to_string(mode));
    }
}

SessionSettings readSessionSettings(const std::string &filepath) {
    auto logger = ::arrus::getDefaultLogger();
    // Read ARRUS_PATH.
    const char *arrusPathStr = std::getenv(ARRUS_PATH_KEY);
    boost::filesystem::path arrusPath;
    if (arrusPathStr != nullptr) {
        arrusPath = arrusPathStr;
    }
    // Read and validate session.
    boost::filesystem::path sessionSettingsPath{filepath};
    // Try with the provided path first.
    if (!boost::filesystem::is_regular_file(sessionSettingsPath)) {
        // Next, try with ARRUS_PATH.
        if (!arrusPath.empty() && sessionSettingsPath.is_relative()) {
            sessionSettingsPath = arrusPath / sessionSettingsPath;
            if (!boost::filesystem::is_regular_file(sessionSettingsPath)) {
                throw IllegalArgumentException(::arrus::format("File not found {}.", filepath));
            }
        } else {
            throw IllegalArgumentException(::arrus::format("File not found {}.", filepath));
        }
    }

    std::string settingsPathStr = sessionSettingsPath.string();
    logger->log(LogSeverity::INFO, ::arrus::format("Using configuration file: {}", settingsPathStr));

    std::unique_ptr<ap::SessionSettings> s = readProtoTxt<ap::SessionSettings>(settingsPathStr);
    //Validate.
    SessionSettingsProtoValidator validator("session settings in " + settingsPathStr);
    validator.validate(s);
    validator.throwOnErrors();

    // Read and validate Dictionary.
    std::unique_ptr<ap::Dictionary> d;
    if (!s->dictionary_file().empty()) {
        std::string dictionaryPathStr;
        // 1. Try to find the file relative to the current working directory.
        if (boost::filesystem::is_regular_file(s->dictionary_file())) {
            dictionaryPathStr = s->dictionary_file();
        } else {
            // 2. Try to use the parent directory of session settings.
            auto dictP = sessionSettingsPath.parent_path() / s->dictionary_file();
            if (boost::filesystem::is_regular_file(dictP)) {
                dictionaryPathStr = dictP.string();
            } else {
                // 3. Try to use ARRUS_PATH, if available.
                if (!arrusPath.empty()) {
                    boost::filesystem::path arrusDicP = arrusPath / s->dictionary_file();
                    if (boost::filesystem::is_regular_file(arrusDicP)) {
                        dictionaryPathStr = arrusDicP.string();
                    } else {
                        throw IllegalArgumentException(
                            ::arrus::format("Invalid path to dictionary: {}", s->dictionary_file()));
                    }
                } else {
                    throw IllegalArgumentException(
                        ::arrus::format("Invalid path to dictionary: {}", s->dictionary_file()));
                }
            }
        }
        d = readProtoTxt<ap::Dictionary>(dictionaryPathStr);
        logger->log(LogSeverity::INFO, ::arrus::format("Using dictionary file: {}", dictionaryPathStr));
    } else {
        // Read default dictionary.
        try {
            d = readProtoTxtStr<ap::Dictionary>(arrus::io::DEFAULT_DICT);
        } catch (const IllegalArgumentException &e) {
            throw IllegalArgumentException(::arrus::format("Error while reading ARRUS default "
                                                           "dictionary. Message: {}",
                                                           e.what()));
        }
        logger->log(LogSeverity::INFO, "Using default dictionary.");
    }
    DictionaryProtoValidator dictionaryValidator("dictionary");
    dictionaryValidator.validate(d);
    dictionaryValidator.throwOnErrors();

    SettingsDictionary dictionary = readDictionary(d.get());
    SessionSettingsBuilder settingsBuilder;
    if (s->has_us4r()) {
        settingsBuilder.addUs4R(readUs4RSettings(s->us4r(), dictionary));
    }
    if (s->has_file()) {
        settingsBuilder.addFile(readFileSettings(s->file(), dictionary));
    }
    SessionSettings settings = settingsBuilder.build();
    logger->log(LogSeverity::DEBUG, arrus::format("Read settings from '{}': {}", filepath, arrus::toString(settings)));
    return settings;
}

}// namespace arrus::io