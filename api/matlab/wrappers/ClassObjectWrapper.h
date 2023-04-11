#ifndef API_MATLAB_WRAPPERS_CLASSOBJECTWRAPPER_H
#define API_MATLAB_WRAPPERS_CLASSOBJECTWRAPPER_H

#include <ostream>
#include <string>
#include <utility>

#include "api/matlab/wrappers/Ptr.h"
#include "api/matlab/wrappers/asserts.h"
#include "api/matlab/wrappers/MatlabClassImpl.h"

#include "arrus/common/format.h"
#include "arrus/core/api/devices/DeviceId.h"
#include "arrus/core/api/session/Session.h"
#include "arrus/core/api/session/SessionSettings.h"

namespace arrus::matlab {

template<typename T> class ClassObjectWrapper : public MatlabClassImpl {
public:
    using MatlabClassImpl::MatlabClassImpl;

    typedef T ValueType;

    /**
     * Creates a new object. In the case of simple pointer wrapper, this is NOP.
     */
    MatlabObjectHandle create(std::shared_ptr<MexContext> ctx, MatlabInputArgs &args) override {
        throw ::arrus::IllegalArgumentException(format("This class of objects: {} cannot be instantiated by "
                                                       "arrus MATLAB API, probably there is some other method "
                                                       "to get that value...", getClassId()));
    };

    /**
     * Deletes a given object. In the case of simply pointer wrappers, the object is not managed
     * by this implementation, so do nothing.
     *
     * @param handle a handle to the deleted object
     */
    void remove(const MatlabObjectHandle handle) override {
        throw ::arrus::IllegalArgumentException(format("This class of objects: {} cannot be removed by "
                                                       "arrus MATLAB API, probably this object is managed by some"
                                                       "other mechanism. ", getClassId()));
    }

protected:
    /**
     * Simply, assume that the provided value is a correct pointer.
     */
    T *get(MatlabObjectHandle handle) {
        return reinterpret_cast<T*>(handle);
    }
};

}

#endif//API_MATLAB_WRAPPERS_CLASSOBJECTWRAPPER_H
