#ifndef ARRUS_API_MATLAB_WRAPPERS_MEXOBJECTMANAGER_H
#define ARRUS_API_MATLAB_WRAPPERS_MEXOBJECTMANAGER_H

#include <memory>
#include <unordered_map>
#include <utility>

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/MexObjectWrapper.h"
#include "api/matlab/wrappers/common.h"
#include "arrus/common/asserts.h"
#include "arrus/common/format.h"

namespace arrus::matlab {

class MexObjectManager {
public:
    using MexObjectPtr = std::unique_ptr<MexObjectWrapper>;

    explicit MexObjectManager(std::shared_ptr<MexContext> ctx, MexObjectClassId classId)
        : classId(std::move(classId)), ctx(std::move(ctx)) {}

    virtual MexObjectHandle create(std::shared_ptr<MexContext> ctx, MexMethodArgs &args) = 0;

    virtual void remove(const MexObjectHandle handle) { objects.erase(handle); }

    MexObjectPtr &getObject(const MexObjectHandle handle) { return objects.at(handle); }

protected:
    std::shared_ptr<MexContext> ctx;

    /**
     * Assigns new unique handle to the object and stores in the underlying
     * map.
     */
    MexObjectHandle insert(MexObjectPtr obj) {
        std::unique_lock<std::mutex> lock{objectHandleMutex};
        // lastHandle modification requires exclusive access
        MexObjectHandle handle = lastHandle++;
        auto res = objects.insert(std::make_pair(handle, std::move(obj)));
        ARRUS_REQUIRES_TRUE(res.second,
                            "Mex object manager internal error: could not store "
                            "newly created object");
        return handle;
    }

private:
    MexObjectClassId classId;
    std::unordered_map<MexObjectHandle, MexObjectPtr> objects;
    MexObjectHandle lastHandle{0};
    std::mutex objectHandleMutex;
};

}// namespace arrus::matlab

#endif//ARRUS_API_MATLAB_WRAPPERS_MEXOBJECTMANAGER_H
