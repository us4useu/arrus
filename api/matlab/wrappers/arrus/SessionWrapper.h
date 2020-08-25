#ifndef ARRUS_API_MATLAB_WRAPPERS_ARRUS_SESSIONOBJECTWRAPPER_H
#define ARRUS_API_MATLAB_WRAPPERS_ARRUS_SESSIONOBJECTWRAPPER_H

#include <utility>
#include <string>

#include "arrus/api/matlab/wrappers/MexObjectWrapper.h"
#include "arrus/api/matlab/wrappers/DefaultMexObjectManager.h"

#include "arrus/core/api/session/Session.h"
#include "arrus/common/format.h"


namespace arrus::matlab {


class SessionWrapper : public MexObjectWrapper {
public:

    explicit SessionWrapper(
            std::shared_ptr<MexContext> ctx,
            const MexMethodArgs &args)
            : MexObjectWrapper(std::move(ctx)) {
        this->ctx->logInfo("Constructor");

        // TODO remove
        this->ctx->logInfo("calling");
        auto value = ::arrus::testConnection();
        this->ctx->logInfo("after calling");
        this->ctx->logInfo(
            arrus::format("Test connection, {}", value)
        );
        this->ctx->logInfo("1");

        this->addMethod("test1", std::bind(&SessionWrapper::test1,
                this, std::placeholders::_1));
        this->ctx->logInfo("2");
        this->addMethod("test2", std::bind(&SessionWrapper::test2,
                this, std::placeholders::_1));
        this->ctx->logInfo("3");
    }

    ~SessionWrapper() override {
        ctx->logInfo("Destructor");
    }

    MexMethodReturnType test1(const MexMethodArgs &inputs) {
        return ctx->getArrayFactory().createCellArray({0});
    }

    MexMethodReturnType test2(const MexMethodArgs &inputs) {
        return ctx->getArrayFactory().createCellArray({1, 2},
                "abc",
                ctx->getArrayFactory().createArray<double>({2, 2}, {1., 2., 3., 4.}));
    }
};


class SessionWrapperManager : public DefaultMexObjectManager<SessionWrapper> {
    using DefaultMexObjectManager::DefaultMexObjectManager;
};

}

#endif
