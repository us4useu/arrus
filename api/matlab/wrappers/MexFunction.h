#ifndef MEXOBJECTFUNCTION_H
#define MEXOBJECTFUNCTION_H

#include <unordered_map>
#include <string>
#include <memory>

#include "arrus/common/compiler.h"
#include "arrus/common/asserts.h"
#include "api/matlab/wrappers/common.h"
#include "api/matlab/wrappers/MexObjectManager.h"
#include "api/matlab/wrappers/MexObjectWrapper.h"
#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/arrus/SessionWrapper.h"

COMPILER_PUSH_DIAGNOSTIC_STATE
#pragma warning(disable: 4100 4189 4458 4702)
#include <mex.hpp>
#include <mexAdapter.hpp>
COMPILER_POP_DIAGNOSTIC_STATE

using namespace arrus::matlab;
using namespace matlab::mex;

/**
 *
 * This class is responsible for:
 * - managing MexObjectManagers (stores map: class id -> MexObjectClassManager)
 * - translating input arguments list to class, method, object handle, other parameters
 * - handling all exceptions that happen durring method invocation
 */
class MexFunction : public matlab::mex::Function {
public:
    MexFunction();

    ~MexFunction() override;

    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override;

private:
    using ManagerPtr = std::unique_ptr<MexObjectManager>;
    std::shared_ptr<MexContext> mexContext{new MexContext(getEngine())};

    std::unordered_map<MexObjectClassId, ManagerPtr> managers;
};


#endif // !MEXOBJECTFUNCTION_H

