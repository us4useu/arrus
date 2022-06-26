#ifndef ARRUS_API_MATLAB_WRAPPERS_MATLABSTDOUTBUFFER_H
#define ARRUS_API_MATLAB_WRAPPERS_MATLABSTDOUTBUFFER_H

#include "mex.hpp"
#include "mexAdapter.hpp"
#include <sstream>
#include <utility>

namespace arrus::matlab {

class MatlabStdoutBuffer : public std::stringbuf {
public:
    explicit MatlabStdoutBuffer(std::shared_ptr<::matlab::engine::MATLABEngine> matlabEngine)
        : basic_stringbuf(std::ios_base::out), matlabEngine(std::move(matlabEngine)) {}

    int sync() override {
        std::cout << this->str() << std::endl;
//        matlabEngine->fevalAsync(u"fprintf", 0, std::vector<::matlab::data::Array>({factory.createScalar(this->str())}));
        this->str("");
        return 0;
    }

private:
    ::matlab::data::ArrayFactory factory;
    std::shared_ptr<::matlab::engine::MATLABEngine> matlabEngine;
};

}// namespace arrus::matlab

#endif//ARRUS_API_MATLAB_WRAPPERS_MATLABSTDOUTBUFFER_H
