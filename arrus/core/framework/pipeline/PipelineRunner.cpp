#include <cmath>
#include <fstream>
#include <iostream>
#include <utility>

#include "Metadata.h"
#include "NdArray.h"
#include "PipelineRunner.h"
#include "imaging/KernelRegistry.h"
#include "pwi.h"

namespace arrus_example_imaging {

PipelineRunner::PipelineRunner(NdArrayDef inputDef, std::shared_ptr<Metadata> metadata, Pipeline pipeline)
    : inputDef(std::move(inputDef)), inputMetadata(std::move(metadata)), pipeline(std::move(pipeline)) {
    inputGpu = NdArray{this->inputDef, true};
    CUDA_ASSERT(cudaStreamCreate(&processingStream));
    prepare();
}

PipelineRunner::~PipelineRunner() { CUDA_ASSERT_NO_THROW(cudaStreamDestroy(processingStream)); }

void PipelineRunner::prepare() {
    auto &registry = KernelRegistry::getInstance();
    NdArrayDef currentInputDef = inputDef;
    std::shared_ptr<Metadata> currentMetadata = inputMetadata;
    NdArray currentInputArray = inputGpu.createView();

    for (auto &op : pipeline.getOps()) {
        // Create Construction context: getArray the current NdArrayDef (start with inputDef),
        KernelConstructionContext constructionContext{currentInputDef, currentInputDef, currentMetadata, op.getParams()};
        // determine kernel, run factory function from Registry
        // Create output array for that kernel, create context for that
        Kernel::Handle kernel = registry.createKernel(op, constructionContext);
        kernels.push_back(std::move(kernel));
        kernelOutputs.emplace_back(constructionContext.getOutput(), true);
        auto &outputArray = kernelOutputs[kernelOutputs.size() - 1];
        kernelExecutionCtx.emplace_back(currentInputArray, outputArray.createView(), processingStream);
        currentInputDef = constructionContext.getOutput();
        currentMetadata = constructionContext.getOutputMetadataBuilder().buildSharedPtr();
        currentInputArray = outputArray.createView();
    }
    this->outputDef = currentInputDef;
    outputHost = NdArray(this->outputDef, false);
}

void PipelineRunner::process(const ::arrus::framework::BufferElement::SharedHandle &ptr,
                             void (*processingCallback)(void *)) {
    // NOTE: data transfers H2D, D2H and processing are intentionally
    // serialized here into a single 'processingStream', for the sake
    // of simplicity.
    // Normally, n-element buffers should probably be used (with some
    // additional synchronization or overwrite detection) as a common
    // memory area for communication between RF data producer and
    // consumer.

    auto &inputArray = ptr->getData();
    // Wrap pointer to the input data into NdArray object.
    NdArray inputHost{inputArray.get<int16_t>(), inputDef, false};
    //    Transfer data H2D.
    CUDA_ASSERT(cudaMemcpyAsync(inputGpu.getPtr<void>(), inputHost.getPtr<void>(), inputHost.getNBytes(),
                                cudaMemcpyHostToDevice, processingStream));
    // Release host RF buffer element after transferring the data.
    CUDA_ASSERT(cudaLaunchHostFunc(
        processingStream, [](void *element) { ((::arrus::framework::BufferElement *) element)->release(); },
        ptr.get()));

    // Execute a sequence of pipeline kernels.
    for (size_t i = 0; i < kernels.size(); ++i) {
        kernels[i]->process(kernelExecutionCtx[i]);
    }
    auto &pipelineOutput = kernelOutputs[kernelOutputs.size() - 1];
    //    Transfer data D2H.
    CUDA_ASSERT(cudaMemcpyAsync(outputHost.getPtr<void>(), pipelineOutput.getPtr<void>(), outputHost.getNBytes(),
                                cudaMemcpyDeviceToHost, processingStream));
    CUDA_ASSERT(cudaStreamSynchronize(processingStream));
    processingCallback(outputHost.getPtr<void>());
    //    There seems to be some issues when calling opencv::imshow in cuda callback,
    //        so I had to use cudaStreamSynchronize here.
    //    CUDA_ASSERT(cudaLaunchHostFunc(processingStream, processingCallback, outputHost.getPtr<void>()));
}
const NdArrayDef &PipelineRunner::getOutputDef() const { return outputDef; }
}// namespace arrus_example_imaging
