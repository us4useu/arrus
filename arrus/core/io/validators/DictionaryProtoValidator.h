#ifndef ARRUS_CORE_IO_VALIDATORS_DICTIONARYPROTOVALIDATOR_H
#define ARRUS_CORE_IO_VALIDATORS_DICTIONARYPROTOVALIDATOR_H

#include <string>
#include <utility>
#include <unordered_set>
#include <boost/functional/hash.hpp>

#include "arrus/common/compiler.h"
#include "arrus/core/common/collections.h"
#include "arrus/core/common/validation.h"

#include "arrus/core/io/validators/ProbeModelProtoValidator.h"
#include "arrus/core/io/validators/ProbeAdapterModelProtoValidator.h"
#include "arrus/core/io/validators/ProbeToAdapterConnectionProtoValidator.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
COMPILER_DISABLE_MSVC_WARNINGS(4127)

#include "io/proto/Dictionary.pb.h"

COMPILER_POP_DIAGNOSTIC_STATE

namespace arrus::io {

class DictionaryProtoValidator
    : public Validator<std::unique_ptr<arrus::proto::Dictionary>> {

public:
    using Validator::Validator;

    template<typename T>
    inline bool hasId(T model) {
        return model.has_id() && !model.id().name().empty()
               && !model.id().manufacturer().empty();
    }

    void
    validate(const std::unique_ptr<arrus::proto::Dictionary> &obj) override {
        using ModelId = std::pair<std::string, std::string>;
        std::unordered_set<ModelId, boost::hash<ModelId>> adapterIds;
        std::unordered_set<ModelId, boost::hash<ModelId>> probeIds;

        // Validate probes
        int i = 0;
        for(auto &probe : obj->probe_models()) {
            std::string fieldName = arrus::format("probe_model:{}", i);
            expectTrue(fieldName, hasId(probe),
                       "id (including its components) should not empty");
            if(hasId(probe)) {
                ModelId probeId = {probe.id().manufacturer(),
                                   probe.id().name()};

                expectTrue(fieldName,
                           probeIds.find(probeId) == probeIds.end(),
                           "id already used.");

                probeIds.emplace(probeId);

                ProbeModelProtoValidator probeValidator(fieldName);
                probeValidator.validate(probe);
                copyErrorsFrom(probeValidator);
            }
            ++i;
        }

        i = 0;
        for(auto &adapter: obj->probe_adapter_models()) {
            std::string fieldName = arrus::format("probe_adapter_model:{}", i);
            expectTrue(fieldName, hasId(adapter),
                       "id (including its components) should not empty");
            if(hasId(adapter)) {
                ModelId adapterId = {adapter.id().manufacturer(),
                                     adapter.id().name()};
                expectTrue(fieldName,
                           adapterIds.find(adapterId) == adapterIds.end(),
                           "id already used.");

                adapterIds.emplace(adapterId);

                ProbeAdapterModelProtoValidator adapterValidator(fieldName);
                adapterValidator.validate(adapter);
                copyErrorsFrom(adapterValidator);
                ++i;
            }

        }
        i = 0;
        for(auto &conn : obj->probe_to_adapter_connections()) {
            std::string fieldName =
                arrus::format("probe_adapter_connection:{}", i);

            // Verify if the probe model with given id actually exists.
            ModelId probeModelId = {conn.probe_model_id().manufacturer(),
                                    conn.probe_model_id().name()};

            expectTrue(fieldName, probeIds.find(probeModelId) != probeIds.end(),
                       arrus::format("Undefined probe id: {}, {}",
                                     probeModelId.first, probeModelId.second));

            // Verify if the adapter models with given ids actually exist.
            for(auto &probeAdapterModelId: conn.probe_adapter_model_id()) {
                ModelId id = {probeAdapterModelId.manufacturer(),
                              probeAdapterModelId.name()};
                expectTrue(fieldName, adapterIds.find(id) != adapterIds.end(),
                           arrus::format("Undefined adapter id: {}, {}",
                                         id.first, id.second));
            }

            ProbeToAdapterConnectionProtoValidator connValidator(fieldName);
            connValidator.validate(conn);
            copyErrorsFrom(connValidator);

            ++i;
        }
    }

};

}

#endif //ARRUS_CORE_IO_VALIDATORS_DICTIONARYPROTOVALIDATOR_H
