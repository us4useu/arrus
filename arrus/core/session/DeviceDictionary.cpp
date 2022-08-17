#include "arrus/core/api/session/DeviceDictionary.h"

namespace arrus::session {

class DeviceDictionary::Impl {
public:
    [[nodiscard]] ProbeAdapterSettings getAdapterSettings(const ProbeAdapterModelId &adapterModelId) const {
        std::string idStr = convertIdToString(adapterModelId);
        try {
            return adaptersMap.at(idStr);
        } catch (const std::out_of_range &) {
            throw IllegalArgumentException(::arrus::format("Adapter model not found: {}", idStr));
        }
    }

    [[nodiscard]] ProbeSettings getProbeSettings(const ProbeModelId &probeModelId,
                                                 const ProbeAdapterModelId &adapterModelId) const {
        std::string probeModelIdStr = convertIdToString(probeModelId);
        std::string adapterModelIdStr = convertIdToString(adapterModelId);
        std::string key = probeModelIdStr + adapterModelIdStr;
        try {
            return probesMap.at(key);
        } catch (const std::out_of_range &) {
            throw IllegalArgumentException(::arrus::format("There is no defined setting in the dictionary "
                                                           "for pair: ({}, {}), ({}, {})",
                                                           probeModelId.getManufacturer(), probeModelId.getName(),
                                                           adapterModelId.getManufacturer(), adapterModelId.getName()));
        }
    }

    [[nodiscard]] ProbeModel getProbeModel(const ProbeModelId &id) const {
        std::string idStr = convertIdToString(id);
        try {
            return modelsMap.at(idStr);
        } catch (const std::out_of_range &) {
            throw IllegalArgumentException(::arrus::format("Probe model not found: {}", idStr));
        }
    }

    template<typename T> static std::string convertIdToString(const T &id) {
        return id.getManufacturer() + id.getName();
    }

    template<typename T> static std::string convertProtoIdToString(const T &id) {
        return id.manufacturer() + id.name();
    }

private:
    // manufacturer + name -> adapter
    std::unordered_map<std::string, ProbeAdapterSettings> adaptersMap;
    // adapter manufacturer + a. name + probe manufacturer + p. name -> probe s.
    std::unordered_map<std::string, ProbeSettings> probesMap;
    std::unordered_map<std::string, ProbeModel> modelsMap;
    friend class DeviceDictionaryBuilder;
};

DeviceDictionary::DeviceDictionary(const DeviceDictionary &o) = default;
DeviceDictionary::DeviceDictionary(DeviceDictionary &&o) noexcept = default;
DeviceDictionary::~DeviceDictionary() = default;
DeviceDictionary &DeviceDictionary::operator=(const DeviceDictionary &o) = default;
DeviceDictionary &DeviceDictionary::operator=(DeviceDictionary &&o) noexcept = default;

ProbeAdapterSettings DeviceDictionary::getAdapterSettings(const ProbeAdapterModelId &adapterModelId) const {
    return impl->getAdapterSettings(adapterModelId);
}
ProbeSettings DeviceDictionary::getProbeSettings(const ProbeModelId &probeId,
                                                 const ProbeAdapterModelId &adapterId) const {
    return impl->getProbeSettings(probeId, adapterId);
}
ProbeModel DeviceDictionary::getProbeModel(const ProbeModelId &id) const { return impl->getProbeModel(id); }

class DeviceDictionaryBuilder::Impl {
public:
    void insertAdapterSettings(ProbeAdapterSettings &&adapter) {
        std::string key = DeviceDictionary::Impl::convertIdToString(adapter.getModelId());
        dict.adaptersMap.emplace(key, std::forward<ProbeAdapterSettings>(adapter));
    }

    void insertProbeSettings(ProbeSettings &&probe, const ProbeAdapterModelId &adapterId) {
        std::string adapterKey = DeviceDictionary::Impl::convertIdToString(adapterId);
        std::string probeKey = DeviceDictionary::Impl::convertIdToString(probe.getModel().getModelId());
        std::string key = probeKey + adapterKey;
        dict.probesMap.emplace(key, std::forward<ProbeSettings>(probe));
    }

    void insertProbeModel(const ProbeModel &probeModel) {
        std::string key = DeviceDictionary::Impl::convertIdToString(probeModel.getModelId());
        dict.modelsMap.emplace(key, probeModel);
    }

private:
    DeviceDictionary::Impl dict;
};
DeviceDictionaryBuilder &DeviceDictionaryBuilder::addAdapterModel(ProbeAdapterSettings adapter) {
    impl->insertAdapterSettings(std::move(adapter));
    return *this;
}
DeviceDictionaryBuilder &DeviceDictionaryBuilder::addProbeSettings(ProbeSettings probe,
                                                                   const ProbeAdapterModelId &adapterId) {
    impl->insertProbeSettings(std::move(probe), adapterId);
    return *this;
}

DeviceDictionaryBuilder &DeviceDictionaryBuilder::addProbeModel(const ProbeModel &probeModel) {
    impl->insertProbeModel(probeModel);
    return *this;
}

DeviceDictionaryBuilder::DeviceDictionaryBuilder(const DeviceDictionaryBuilder &o) = default;
DeviceDictionaryBuilder::DeviceDictionaryBuilder(DeviceDictionaryBuilder &&o) noexcept = default;
DeviceDictionaryBuilder::~DeviceDictionaryBuilder() = default;
DeviceDictionaryBuilder &DeviceDictionaryBuilder::operator=(const DeviceDictionaryBuilder &o) = default;
DeviceDictionaryBuilder &DeviceDictionaryBuilder::operator=(DeviceDictionaryBuilder &&o) noexcept = default;

}// namespace arrus::session
