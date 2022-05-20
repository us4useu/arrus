#ifndef ARRUS_API_MATLAB_WRAPPERS_MATLABOUTBUFFER_H
#define ARRUS_API_MATLAB_WRAPPERS_MATLABOUTBUFFER_H

#include "mex.hpp"
#include "mexAdapter.hpp"
#include <sstream>
#include <utility>

namespace arrus::matlab {

class MatlabOutBuffer : public std::stringbuf {
public:
    explicit MatlabOutBuffer(std::shared_ptr<::matlab::engine::MATLABEngine> matlabEngine)
        : basic_stringbuf(std::ios_base::out), matlabEngine(std::move(matlabEngine)) {}

    int sync() override {
        matlabEngine->feval(u"fprintf", 0, std::vector<::matlab::data::Array>({factory.createScalar(this->str())}));
        this->str("");
        return 0;
    }

private:
    ::matlab::data::ArrayFactory factory;
    std::shared_ptr<::matlab::engine::MATLABEngine> matlabEngine;
};

}// namespace arrus::matlab

#endif//ARRUS_API_MATLAB_WRAPPERS_MATLABOUTBUFFER_H
