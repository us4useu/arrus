#ifndef ARRUS_ARRUS_CORE_API_OPS_OP_H
#define ARRUS_ARRUS_CORE_API_OPS_OP_H
#include <memory>

namespace arrus::ops {

class Op {
public:
    typedef std::shared_ptr<Op> SharedHandle;

//    explicit Op(::arrus::devices::Device *placement) : placement(placement) {}

    virtual unsigned getTypeId() = 0;

private:
//    arrus::devices::Device::RawHandle placement;
};

}

#endif //ARRUS_ARRUS_CORE_API_OPS_OP_H
