#ifndef ARRUS_CORE_IO_SETTINGSDICTIONARY_H
#define ARRUS_CORE_IO_SETTINGSDICTIONARY_H

#include <unordered_map>
#include "arrus/core/api/devices/us4r/ProbeAdapterSettings.h"

namespace arrus::io {

using ::arrus::devices::ProbeAdapterSettings;
using ::arrus::devices::ProbeSettings;
using ::arrus::devices::ProbeModel;
using ::arrus::devices::ProbeModelId;
using ::arrus::devices::ProbeAdapterModelId;

class SettingsDictionary {
    public:
    [[nodiscard]] ProbeAdapterSettings
    getAdapterSettings(const ProbeAdapterModelId &adapterModelId) const {
        std::string idStr = convertIdToString(adapterModelId);
        try {
            return adaptersMap.at(idStr);
        } catch(const std::out_of_range &) {
            throw IllegalArgumentException(::arrus::format("Adapter model not found: {}", idStr));
        }
    }

    void insertAdapterSettings(ProbeAdapterSettings &&adapter) {
        std::string key = convertIdToString(adapter.getModelId());
        adaptersMap.emplace(key,
                            std::forward<ProbeAdapterSettings>(adapter));
    }

    [[nodiscard]] ProbeSettings
    getProbeSettings(const ProbeModelId &probeModelId,
                     const ProbeAdapterModelId &adapterModelId) const {
        std::string probeModelIdStr = convertIdToString(probeModelId);
        std::string adapterModelIdStr = convertIdToString(adapterModelId);
        std::string key = probeModelIdStr + adapterModelIdStr;
        try {
            return probesMap.at(key);
        } catch(const std::out_of_range &) {
            throw IllegalArgumentException(::arrus::format(
                "There is no defined setting in the dictionary "
                "for pair: ({}, {}), ({}, {})",
                probeModelId.getManufacturer(), probeModelId.getName(),
                adapterModelId.getManufacturer(), adapterModelId.getName()));
        }
        return probesMap.at(key);
    }

    void insertProbeSettings(ProbeSettings &&probe,
                             const ProbeAdapterModelId &adapterId) {
        std::string key =
            convertIdToString(probe.getModel().getModelId()) +
            convertIdToString(adapterId);
        probesMap.emplace(key, std::forward<ProbeSettings>(probe));
    }

    [[nodiscard]] ProbeModel getProbeModel(const ProbeModelId &id) const {
        std::string idStr = convertIdToString(id);
        try {
            return modelsMap.at(idStr);
        }
        catch(const std::out_of_range &) {
            throw IllegalArgumentException(::arrus::format("Probe model not found: {}", idStr));
        }
    }

    void insertProbeModel(const ProbeModel &probeModel) {
        std::string key = convertIdToString(probeModel.getModelId());
        modelsMap.emplace(key, probeModel);
    }

    template<typename T>
    static
    std::string convertIdToString(const T &id) {
        return id.getManufacturer() + id.getName();
    }

    template<typename T>
    static
    std::string convertProtoIdToString(const T &id) {
        return id.manufacturer() + id.name();
    }

    private:
    //
    // manufacturer + name -> adapter
    std::unordered_map<std::string, ProbeAdapterSettings> adaptersMap;
    // adapter manufacturer + a. name + probe manufacturer + p. name -> probe s.
    std::unordered_map<std::string, ProbeSettings> probesMap;
    std::unordered_map<std::string, ProbeModel> modelsMap;
};

}

#endif //ARRUS_CORE_IO_SETTINGSDICTIONARY_H
