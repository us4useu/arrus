#ifndef CPP_EXAMPLE_KERNELS_KERNELINITCONTEXT_H
#define CPP_EXAMPLE_KERNELS_KERNELINITCONTEXT_H

#include <utility>

#include "NdArray.h"
#include "imaging/Operation.h"
#include "imaging/Metadata.h"

namespace arrus_example_imaging {
class KernelConstructionContext {
public:
    KernelConstructionContext(NdArrayDef input, NdArrayDef output, std::shared_ptr<Metadata> inputMetadata,
                              OpParameters parameters)
        : input(std::move(input)), output(std::move(output)), inputMetadata(std::move(inputMetadata)),
          parameters(std::move(parameters)){

        // Start with the input metadata.
        outputMetadataBuilder = MetadataBuilder{this->inputMetadata};
    }

    const NdArrayDef &getInput() const { return input; }

    const NdArrayDef &getOutput() const { return output; }

    const NdArray &getParamArray(const std::string &key) const {return parameters.getArray(key);}

    void setOutput(const NdArrayDef &value) {this->output = value; }

    const std::shared_ptr<Metadata> &getInputMetadata() const { return inputMetadata; }

    MetadataBuilder &getOutputMetadataBuilder() { return outputMetadataBuilder; }

private:
    NdArrayDef input, output;
    std::shared_ptr<Metadata> inputMetadata;
    MetadataBuilder outputMetadataBuilder;
    OpParameters parameters;
};
}// namespace arrus_example_imaging
#endif//CPP_EXAMPLE_KERNELS_KERNELINITCONTEXT_H
