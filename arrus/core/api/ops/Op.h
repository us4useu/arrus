#ifndef ARRUS_ARRUS_CORE_API_OPS_OP_H
#define ARRUS_ARRUS_CORE_API_OPS_OP_H
#include <memory>

namespace arrus::ops {

class Op {
public:
    typedef std::shared_ptr<Op> SharedHandle;

//    explicit Op(::arrus::devices::Device *placement) : placement(placement) {}

private:
    std::string typeName;
    arrus::devices::DeviceId placement;
    std::vector<> inputs;
};

}

#endif //ARRUS_ARRUS_CORE_API_OPS_OP_H
