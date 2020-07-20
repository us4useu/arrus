#include "MexObjectFunction.h"

MexFunction::MexFunction() {
    mexLock();
}

MexFunction::~MexFunction() {
    mexUnlock();
}

void MexFunction::operator()(matlab::mex::ArgumentList outputs,
                             matlab::mex::ArgumentList inputs) {
    //
    outputs[0] = factory.createArray<double>({3}, {1, 2, 3});
}

