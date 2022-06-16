#ifndef CPP_EXAMPLE_IMAGING_OPS_PIPELINE_H
#define CPP_EXAMPLE_IMAGING_OPS_PIPELINE_H

#include <utility>
#include <vector>
#include "imaging/Operation.h"

namespace arrus_example_imaging {
class Pipeline {
public:
    explicit Pipeline(std::vector<Operation> ops) : ops(std::move(ops)) {}

    [[nodiscard]] const std::vector<Operation> &getOps() const { return ops; }

private:
    std::vector<Operation> ops;
};
}

#endif//CPP_EXAMPLE_IMAGING_OPS_PIPELINE_H
