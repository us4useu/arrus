#ifndef MEXFUNCTION_H
#define MEXFUNCTION_H

#include "core/utils/compiler.h"

#pragma warning(push, 0)
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
COMPILER_PUSH_DIAGNOSTIC_STATE
#pragma warning(disable: 4100 4189 4458 4702)

#include <mex.hpp>
#include <mexAdapter.hpp>
COMPILER_POP_DIAGNOSTIC_STATE

class MexFunction : public matlab::mex::Function {
public:
    MexFunction();

    ~MexFunction();

    void operator()(matlab::mex::ArgumentList outputs,
                    matlab::mex::ArgumentList inputs) override;

private:
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
    matlab::data::ArrayFactory factory;
};

#endif // !MEXFUNCTION_H

