#ifndef ARRUS_CORE_SESSION_SETTINGS_DICTIONARY_H
#define ARRUS_CORE_SESSION_SETTINGS_DICTIONARY_H

#include "arrus/common/format.h"
#include "arrus/core/api/devices.h"
#include <unordered_map>

namespace arrus::session {

using arrus::devices::ProbeAdapterSettings;
using arrus::devices::ProbeAdapterModelId;
using arrus::devices::ProbeSettings;
using arrus::devices::ProbeModelId;
using arrus::devices::ProbeModel;

class DeviceDictionaryBuilder;

class DeviceDictionary {
    class Impl;
    UniqueHandle<Impl> impl;
    friend class DeviceDictionaryBuilder;
public:
    ProbeAdapterSettings getAdapterSettings(const ProbeAdapterModelId &adapterModelId) const;
    ProbeSettings getProbeSettings(const ProbeModelId &probeId, const ProbeAdapterModelId &adapterId) const;
    ProbeModel getProbeModel(const ProbeModelId &id) const;

    DeviceDictionary(const DeviceDictionary &o);
    DeviceDictionary(DeviceDictionary &&o) noexcept;
    virtual ~DeviceDictionary();
    DeviceDictionary& operator=(const DeviceDictionary &o);
    DeviceDictionary& operator=(DeviceDictionary &&o) noexcept;
};

class DeviceDictionaryBuilder {
    class Impl;
    UniqueHandle<Impl> impl;
public:
    DeviceDictionaryBuilder &addAdapterModel(ProbeAdapterSettings adapter);
    DeviceDictionaryBuilder &addProbeSettings(ProbeSettings probe, const ProbeAdapterModelId &adapterId);
    DeviceDictionaryBuilder &addProbeModel(const ProbeModel &probeModel);
    DeviceDictionary build();

    DeviceDictionaryBuilder(const DeviceDictionaryBuilder &o);
    DeviceDictionaryBuilder(DeviceDictionaryBuilder &&o) noexcept;
    virtual ~DeviceDictionaryBuilder();
    DeviceDictionaryBuilder& operator=(const DeviceDictionaryBuilder &o);
    DeviceDictionaryBuilder& operator=(DeviceDictionaryBuilder &&o) noexcept;
};


}// namespace arrus::devices

#endif//ARRUS_CORE_SESSION_SETTINGS_DICTIONARY_H
