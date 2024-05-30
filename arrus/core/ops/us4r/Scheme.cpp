#include <utility>

#include "arrus/core/api/ops/us4r/Scheme.h"
#include "arrus/common/format.h"


namespace arrus::ops::us4r {

using namespace arrus::framework;

// Impl.
class Scheme::Impl {
public:
    Impl() = default;

    Impl(std::vector<TxRxSequence> txRxSequences, uint16 rxBufferSize,
         const DataBufferSpec &outputBuffer, WorkMode workMode,
         std::optional<DigitalDownConversion> ddc, const std::vector<NdArray> &constants)
        : txRxSequences(std::move(txRxSequences)), rxBufferSize(rxBufferSize), outputBuffer(outputBuffer), workMode(workMode),
          ddc(std::move(ddc)), constants(constants) {}

    [[nodiscard]] const TxRxSequence &getTxRxSequence() const {
        return getTxRxSequence(0);
    }

    [[nodiscard]] const TxRxSequence &getTxRxSequence(size_t ordinal) const {
        if(ordinal >= txRxSequences.size()) {
            throw IllegalArgumentException(format("Exceeded the maximum number of sequences: {}", txRxSequences.size()));
        }
        return txRxSequences.at(0);
    }
    [[nodiscard]] size_t getNumberOfSequences() const {
        return txRxSequences.size();
    }
    [[nodiscard]] uint16 getRxBufferSize() const { return rxBufferSize; }
    [[nodiscard]] const DataBufferSpec &getOutputBuffer() const { return outputBuffer; }
    [[nodiscard]] WorkMode getWorkMode() const { return workMode; }
    [[nodiscard]] const std::optional<DigitalDownConversion> &getDigitalDownConversion() const { return ddc; }
    [[nodiscard]] const std::vector<NdArray> &getConstants() const { return constants; }
    std::vector<TxRxSequence> const &getTxRxSequences() const { return txRxSequences; }

private:
    friend class SchemeBuilder;
    std::vector<TxRxSequence> txRxSequences;
    uint16 rxBufferSize{2};
    DataBufferSpec outputBuffer;
    WorkMode workMode{WorkMode::HOST};
    std::optional<DigitalDownConversion> ddc;
    std::vector<NdArray> constants;
};

Scheme::Scheme() {
    this->impl = UniqueHandle<Impl>::create();
}

// Scheme
Scheme::Scheme(TxRxSequence txRxSequence, uint16 rxBufferSize, const DataBufferSpec &outputBuffer, WorkMode workMode,
               std::optional<DigitalDownConversion> ddc, const std::vector<NdArray> &constants) {
    std::vector<TxRxSequence> sequences = {std::move(txRxSequence)};
    this->impl = UniqueHandle<Impl>::create(
        std::move(sequences), rxBufferSize, outputBuffer, workMode, std::move(ddc), constants
    );
}
Scheme::Scheme(const Scheme &o) = default;
Scheme::Scheme(Scheme &&o)  noexcept = default;
Scheme::~Scheme() = default;
Scheme &Scheme::operator=(const Scheme &o) = default;
Scheme &Scheme::operator=(Scheme &&o) noexcept = default;

const TxRxSequence & Scheme::getTxRxSequence() const {return impl->getTxRxSequence(); }
const TxRxSequence & Scheme::getTxRxSequence(size_t ordinal) const {return impl->getTxRxSequence(ordinal); }
const std::vector<TxRxSequence> &Scheme::getTxRxSequences() const {return impl->getTxRxSequences();}
const std::vector<NdArray> & Scheme::getConstants() const {return impl->getConstants(); }
const std::optional<DigitalDownConversion> & Scheme::getDigitalDownConversion() const { return impl->getDigitalDownConversion(); }
Scheme::WorkMode Scheme::getWorkMode() const { return impl->getWorkMode(); }
uint16 Scheme::getRxBufferSize() const { return impl->getRxBufferSize(); }
const framework::DataBufferSpec & Scheme::getOutputBuffer() const {return impl->getOutputBuffer(); }

// Builder
Scheme SchemeBuilder::build() {
    Scheme result = std::move(this->scheme);
    this->scheme = Scheme{};
    return result;
}

SchemeBuilder & SchemeBuilder::addSequence(TxRxSequence sequence) {
    this->scheme.impl->txRxSequences.push_back(std::move(sequence));
    return *this;
}

SchemeBuilder& SchemeBuilder::withOutputBufferDefinition(DataBufferSpec spec) {
    this->scheme.impl->outputBuffer = spec;
    return *this;
}

SchemeBuilder& SchemeBuilder::withRxBufferSize(uint16 size) {
    this->scheme.impl->rxBufferSize = size;
    return *this;
}

SchemeBuilder& SchemeBuilder::withWorkMode(Scheme::WorkMode mode) {
    this->scheme.impl->workMode = mode;
    return *this;
}

SchemeBuilder& SchemeBuilder::withDigitalDownConversion(DigitalDownConversion ddc) {
    this->scheme.impl->ddc = std::move(ddc);
    return *this;
}


}