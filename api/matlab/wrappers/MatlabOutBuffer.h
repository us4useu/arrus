#ifndef ARRUS_API_MATLAB_WRAPPERS_MATLABOUTBUFFER_H
#define ARRUS_API_MATLAB_WRAPPERS_MATLABOUTBUFFER_H

#include <sstream>
#include <utility>
#include "mex.hpp"
#include "mexAdapter.hpp"

namespace arrus::matlab {

class MatlabOutBuffer : public std::stringbuf {
public:
    explicit MatlabOutBuffer(
        std::shared_ptr<::matlab::engine::MATLABEngine> matlabEngine)
        : basic_stringbuf(std::ios_base::out),
          matlabEngine(std::move(matlabEngine)) {}

    int sync() override {
        matlabEngine->feval(u"fprintf", 0,
                            std::vector<::matlab::data::Array>(
                                {factory.createScalar(this->str())}));
        this->str("");
        return 0;
    }


private:
    ::matlab::data::ArrayFactory factory;
    std::shared_ptr<::matlab::engine::MATLABEngine> matlabEngine;
};

}


#endif //ARRUS_API_MATLAB_WRAPPERS_MATLABOUTBUFFER_H
