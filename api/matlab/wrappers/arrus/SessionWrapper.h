#ifndef ARRUS_API_MATLAB_WRAPPERS_ARRUS_SESSIONOBJECTWRAPPER_H
#define ARRUS_API_MATLAB_WRAPPERS_ARRUS_SESSIONOBJECTWRAPPER_H

#include <utility>

#include "api/matlab/wrappers/MexObjectWrapper.h"
#include "api/matlab/wrappers/DefaultMexObjectManager.h"
#include "core/session/Session.h"

namespace arrus::matlab_wrappers {


class SessionWrapper : public MexObjectWrapper {
public:

    explicit SessionWrapper(
            std::shared_ptr<MexContext> ctx,
            const MexMethodArgs &args)
            : MexObjectWrapper(std::move(ctx)) {
        this->ctx->logInfo("Constructor");
    }

    ~SessionWrapper() {
        ctx->logInfo("Destructor");
    }

    void call(const MexObjectMethodId &id, const MexMethodArgs &inputs,
              const MexMethodArgs &outputs) override {
        ctx->logInfo(arrus::format("Calling method: {}", id));
    }

};

class SessionWrapperManager : public DefaultMexObjectManager<SessionWrapper> {
    using DefaultMexObjectManager::DefaultMexObjectManager;
};

}

#endif
