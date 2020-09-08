#ifndef ARRUS_API_MATLAB_WRAPPERS_DEFAULTMEXOBJECTMANAGER_H
#define ARRUS_API_MATLAB_WRAPPERS_DEFAULTMEXOBJECTMANAGER_H

#include "MexObjectManager.h"
#include "common.h"
#include "arrus/api/matlab/wrappers/session/SessionWrapper.h"

namespace arrus::matlab {

template<typename T>
class DefaultMexObjectManager : public MexObjectManager {
    using MexObjectManager::MexObjectManager;

    MexObjectHandle
    create(std::shared_ptr<MexContext> ctx, MexMethodArgs &args) override {
        auto ptr = std::unique_ptr<T>(new T(ctx, args));
        return insert(std::move(ptr));
    }
};

}

#endif //ARRUS_API_MATLAB_WRAPPERS_DEFAULTMEXOBJECTMANAGER_H
