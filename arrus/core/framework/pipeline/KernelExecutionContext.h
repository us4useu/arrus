#ifndef PWI_CPP_EXAMPLE_IMAGING_EXECUTIONCONTEXT_H
#define PWI_CPP_EXAMPLE_IMAGING_EXECUTIONCONTEXT_H

namespace arrus_example_imaging {

class KernelExecutionContext {
public:
    KernelExecutionContext(const NdArray &input, const NdArray &output, const cudaStream_t stream)
        : input(input), output(output), stream(stream) {}

    const NdArray &getInput() const { return input; }
    NdArray &getOutput() { return output; }
    cudaStream_t getStream() { return stream; }

private:
    // View
    NdArray input;
    // View
    NdArray output;
    cudaStream_t stream;
};

}

#endif //PWI_CPP_EXAMPLE_IMAGING_EXECUTIONCONTEXT_H
