#ifndef CPP_EXAMPLE_PIPELINE_RUNNER_H
#define CPP_EXAMPLE_PIPELINE_H

#include <algorithm>
#include <arrus/core/api/arrus.h>
#include <functional>
#include <utility>
#include <vector>

#include "imaging/DataType.h"
#include "imaging/Kernel.h"
#include "imaging/Operation.h"
#include "imaging/ops/Pipeline.h"
#include "pwi.h"

namespace arrus_example_imaging {

class PipelineRunner {
public:
    explicit PipelineRunner(NdArrayDef inputDef, std::shared_ptr<Metadata> metadata, Pipeline pipeline);

    virtual ~PipelineRunner();

    void process(const ::arrus::framework::BufferElement::SharedHandle &ptr, void (*processingCallback)(void *));

    const NdArrayDef &getOutputDef() const;

private:
    void prepare();

    std::shared_ptr<::arrus::framework::Buffer> inputBuffer;
    std::shared_ptr<Metadata> inputMetadata;
    Pipeline pipeline;

    std::vector<Kernel::Handle> kernels;
    std::vector<KernelExecutionContext> kernelExecutionCtx;
    NdArray inputGpu;
    // Actual arrays.
    std::vector<NdArray> kernelOutputs;
    NdArray outputHost;
    NdArrayDef inputDef, outputDef;
    cudaStream_t processingStream;

};
}// namespace arrus_example_imaging
#endif//CPP_EXAMPLE_PIPELINE_RUNNER_H
