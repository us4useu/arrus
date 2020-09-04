#ifndef ARRUS_CORE_IO_SETTINGSDICTIONARY_H
#define ARRUS_CORE_IO_SETTINGSDICTIONARY_H

#include <unordered_map>

namespace arrus::io {

class SettingsDictionary {
    public:

    [[nodiscard]] ProbeAdapterSettings
    getAdapterSettings(const ProbeAdapterModelId &adapterModelId) const {
        return adaptersMap.at(convertIdToString(adapterModelId));
    }

    void insertAdapterSettings(ProbeAdapterSettings &&adapter) {
        std::string key = convertIdToString(adapter.getModelId());
        adaptersMap.emplace(key,
                            std::forward<ProbeAdapterSettings>(adapter));
    }

    [[nodiscard]] ProbeSettings
    getProbeSettings(const ProbeModelId &probeModelId,
                     const ProbeAdapterModelId &adapterModelId) const {
        std::string key =
            convertIdToString(probeModelId) +
            convertIdToString(adapterModelId);
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
        return modelsMap.at(convertIdToString(id));
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
