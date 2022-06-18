#ifndef ARRUS_CORE_API_FRAMEWORK_PIPELINE_H
#define ARRUS_CORE_API_FRAMEWORK_PIPELINE_H

#include <utility>
#include <vector>

#include "Step.h"

namespace arrus::framework::pipeline {

class Pipeline {
public:
    Pipeline(std::vector<Step::Handle> stages, devices::DeviceId placement)
        : stages(std::move(stages)), placement(std::move(placement)) {}

private:
    std::vector<Step::Handle> stages;
    devices::DeviceId placement;
};

}

#endif//ARRUS_CORE_API_FRAMEWORK_PIPELINE_H
