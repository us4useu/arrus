#ifndef MEXOBJECTFUNCTION_H
#define MEXOBJECTFUNCTION_H

#include <unordered_map>
#include <string>
#include <ostream>
#include <memory>

#include "arrus/common/compiler.h"
#include "arrus/common/asserts.h"
#include "arrus/api/matlab/wrappers/common.h"
#include "arrus/api/matlab/wrappers/MexObjectManager.h"
#include "arrus/api/matlab/wrappers/MexObjectWrapper.h"
#include "arrus/api/matlab/wrappers/MexContext.h"
#include "arrus/api/matlab/wrappers/session/SessionWrapper.h"
#include "arrus/common/logging/impl/Logging.h"

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
    std::unordered_map<MexObjectClassId, ManagerPtr> managers;

    std::shared_ptr<::matlab::engine::MATLABEngine> matlabEngine{getEngine()};
    std::shared_ptr<MatlabOutBuffer> matlabOutBuffer{
        std::make_shared<MatlabOutBuffer>(matlabEngine)};
    std::shared_ptr<std::ostream> matlabOstream{
        std::make_shared<std::ostream>(matlabOutBuffer.get())};
    std::shared_ptr<arrus::Logging> logging;
    std::shared_ptr<MexContext> mexContext{new MexContext(matlabEngine)};

    void setConsoleLogIfNecessary(const arrus::LogSeverity sev);

    arrus::LogSeverity getLoggerSeverity(ArgumentList inputs);
};


#endif // !MEXOBJECTFUNCTION_H

