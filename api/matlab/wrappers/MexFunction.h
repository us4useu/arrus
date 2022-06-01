#ifndef MEXOBJECTFUNCTION_H
#define MEXOBJECTFUNCTION_H

#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>

#include "api/matlab/wrappers/MexContext.h"
#include "api/matlab/wrappers/MatlabClassImpl.h"
#include "api/matlab/wrappers/Ptr.h"
#include "api/matlab/wrappers/common.h"
//#include "api/matlab/wrappers/session/SessionClassImpl.h"
#include "arrus/common/asserts.h"
#include "arrus/common/compiler.h"
#include "arrus/common/logging/impl/Logging.h"
#include "MatlabStdoutBuffer.h"

#include "mex_headers.h"

using namespace arrus::matlab;
using namespace matlab::mex;

/**
 * This class is responsible for:
 * - managing MexObjectManagers (stores map: class id -> MexObjectClassManager)
 * - translating input arguments list to class, method, object handle, other parameters
 * - handling all exceptions that happen mex function invocation
 */
class MexFunction : public matlab::mex::Function {
public:
    MexFunction();

    ~MexFunction() override;

    void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) override;

private:
    using MatlabClassImplPtr = std::unique_ptr<MatlabClassImpl>;
    std::unordered_map<MatlabClassId, MatlabClassImplPtr> classes;

    std::shared_ptr<::matlab::engine::MATLABEngine> matlabEngine{getEngine()};
    std::shared_ptr<MatlabStdoutBuffer> matlabOutBuffer{std::make_shared<MatlabStdoutBuffer>(matlabEngine)};
    std::shared_ptr<std::ostream> matlabOstream{std::make_shared<std::ostream>(matlabOutBuffer.get())};
    std::shared_ptr<arrus::Logging> logging;
    std::shared_ptr<MexContext> ctx{new MexContext(matlabEngine)};

    void setConsoleLogIfNecessary(arrus::LogSeverity sev);

    arrus::LogSeverity getLoggerSeverity(ArgumentList inputs);

    arrus::LogSeverity convertToLogSeverity(const ::matlab::data::Array &severityStr);

    void addClass(std::unique_ptr<MatlabClassImpl> ptr);
};

#endif// !MEXOBJECTFUNCTION_H
