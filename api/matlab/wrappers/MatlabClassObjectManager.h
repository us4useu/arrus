#ifndef ARRUS_API_MATLAB_WRAPPERS_MATLABCLASSOBJECTMANAGER_H
#define ARRUS_API_MATLAB_WRAPPERS_MATLABCLASSOBJECTMANAGER_H

#include "api/matlab/wrappers/MatlabClassImpl.h"

namespace arrus::matlab {

template<typename T>
class MatlabClassObjectManager: public MatlabClassImpl {
public:
    using MatlabClassImpl::MatlabClassImpl;

    typedef T ValueType;

    /**
     * Deletes a given object. The deleted object should have been previously created using the 'create' method.
     * If no such object exists, an IllegalArgumentException will be thrown.
     *
     * @param handle a handle to the deleted object
     */
    void remove(const MatlabObjectHandle handle) override {
        objects.erase(handle);
    }

protected:
    MatlabObjectHandle insert(std::unique_ptr<T> ptr) {
        auto handle = (MatlabObjectHandle)(ptr.get());
        auto res = objects.insert(std::make_pair(handle, std::move(ptr)));
        ARRUS_REQUIRES_TRUE(res.second,
                            "Mex object manager internal error: could not store "
                            "newly created object (an object with the same handle already exist?).");
        return handle;
    }

private:
    std::unordered_map<MatlabObjectHandle, std::unique_ptr<T>> objects;
};


}

#endif//ARRUS_API_MATLAB_WRAPPERS_MATLABCLASSOBJECTMANAGER_H
