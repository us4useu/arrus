#ifndef ARRUS_API_MATLAB_WRAPPERS_MATLABCLASSIMPL_H
#define ARRUS_API_MATLAB_WRAPPERS_MATLABCLASSIMPL_H

#include <memory>
#include <unordered_map>
#include <utility>

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/Ptr.h"
#include "api/matlab/wrappers/common.h"
#include "arrus/common/asserts.h"
#include "arrus/common/format.h"

namespace arrus::matlab {

/**
 * Provides the underlying implementation for a given class of Matlab objects.
 */
class MatlabClassImpl {
public:

    explicit MatlabClassImpl(std::shared_ptr<MexContext> ctx, MatlabClassId classId)
        : classId(std::move(classId)), ctx(std::move(ctx)) {}

    /**
     * Creates a new object of this class. The created object will be managed by implementation, i.e.
     * the object can be deleted later using remove method.
     *
     * @param ctx MEX context
     * @param args constructor parameter
     * @return a handle to new object
     */
    virtual MatlabObjectHandle create(std::shared_ptr<MexContext> ctx, MatlabMethodArgs &args) = 0;

    /**
     * Deletes a given object. The deleted object should have been previously created using the 'create' method.
     * If no such object exists, an IllegalArgumentException will be thrown.
     *
     * @param handle a handle to the deleted object
     */
    virtual void remove(MatlabObjectHandle handle) = 0;

    /**
     * Calls a method for the object pointed by the provided pointer.
     *
     * @param obj a pointer to object
     * @param method method to call
     * @param args method arguments
     * @return method call result
     */
    MatlabMethodReturnType call(MatlabObjectHandle obj, const MatlabMethodId &method, MatlabMethodArgs &args) {
        try {
            auto func = methods.at(method);
            return func(obj, args);
        } catch(const std::out_of_range &e) {
            throw IllegalArgumentException(format("Class {} has no method with name {}.", classId, method));
        }
    }

    [[nodiscard]] const MatlabClassId &getClassId() const { return classId; }

protected:
    std::shared_ptr<MexContext> ctx;

    typedef std::function<MatlabMethodReturnType(MatlabObjectHandle, MatlabMethodArgs&)> MethodImpl;

    void addMethod(const MatlabMethodId &id, const MethodImpl& impl) {
        methods.emplace(id, impl);
    }

private:
    MatlabClassId classId;
    // Class methods.
    typedef std::unordered_map<MatlabMethodId, MethodImpl> MethodMap;
    MethodMap methods;
};

#define ARRUS_MATLAB_ADD_METHOD(methodId, method)                                                                \
    addMethod((methodId), [this](MatlabObjectHandle obj, MatlabMethodArgs &inputs){this->method(obj, inputs);});



}// namespace arrus::matlab

#endif//ARRUS_API_MATLAB_WRAPPERS_MATLABCLASSIMPL_H
