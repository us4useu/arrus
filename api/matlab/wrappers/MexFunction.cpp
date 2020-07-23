#include "MexFunction.h"

MexFunction::MexFunction() {
    mexLock();
    managers.emplace("Session",
                     new SessionWrapperManager(mexContext, "Session"));
}

MexFunction::~MexFunction() {
    mexUnlock();
}

void MexFunction::operator()(ArgumentList outputs, ArgumentList inputs) {
    try {
        ARRUS_REQUIRES_AT_LEAST(inputs.size(), 2,
                                "The class and method name are missing.");

        MexObjectClassId classId = inputs[0][0];
        MexObjectMethodId methodId = inputs[1][0];

        ManagerPtr& manager = managers.at(classId);

        if (methodId == "create") {
            ArgumentList args(inputs.begin() + 2, inputs.end(),
                              inputs.size() - 2);
            auto handle = manager->create(mexContext, args);
            outputs[0] = mexContext->getArrayFactory().createScalar<MexObjectHandle>(handle);
        } else {
            ARRUS_REQUIRES_AT_LEAST(inputs.size(), 3,
                                    "Object handle is missing.");
            MexObjectHandle handle = inputs[2][0];

            if (methodId == "remove") {
                manager->remove(handle);
            } else {
                ArgumentList args(inputs.begin() + 3, inputs.end(),
                                  inputs.size() - 3);

                auto& object = manager->getObject(handle);
                object->call(methodId, args, outputs);
            }
        }
    }
    catch (const std::exception &e) {
        mexContext->logError(e.what());
    }

}

