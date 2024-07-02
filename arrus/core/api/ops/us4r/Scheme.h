#ifndef ARRUS_CORE_API_OPS_US4R_H
#define ARRUS_CORE_API_OPS_US4R_H

#include <utility>

#include "DigitalDownConversion.h"
#include "TxRxSequence.h"
#include "arrus/core/api/framework/DataBufferSpec.h"

namespace arrus::ops::us4r {

/**
 * A scheme to be executed within the session.
 */
class Scheme {
public:
    /**
     * How the scheme should be executed on the us4r-lite device.
     *
     * This enum value determines the source of the signal trigger
     * (i.e. whether the signal is triggered by us4oem modules or
     * host PC).
     */
    enum class WorkMode {
        /** Trigger generated by us4r, error on overflow. */
        ASYNC,
        /** Trigger generated by us4r, us4r waits on overflow */
        SYNC,
        /** Trigger generated by host, no error on overflow. DEPRECATED: will be replaced in the future by MANUAL mode */
        HOST,
        /** New data acquisition and processing is manually triggered by user. */
        MANUAL
    };

    /**
     * Scheme constructor. This scheme turns off hardware IQ demodulator.
     *
     * @param txRxSequence tx/rx sequence to perform
     * @param rxBufferSize the size of the data acquisition buffer in the memory of the Us4R device
     *   (a single element of the buffer is an output of a single tx/rx sequence execution)
     * @param outputBuffer output buffer specification
     * @param workMode scheme work mode
     */

    Scheme(TxRxSequence txRxSequence, uint16 rxBufferSize, const framework::DataBufferSpec &outputBuffer,
           WorkMode workMode)
        : txRxSequence(std::move(txRxSequence)), rxBufferSize(rxBufferSize), outputBuffer(outputBuffer),
          workMode(workMode), ddc(std::nullopt) {}

    /**
     * Scheme constructor. This scheme turns on hardware IQ demodulator (sees digital down conversion parameter).
     *
     * @param txRxSequence tx/rx sequence to perform
     * @param rxBufferSize the size of the data acquisition buffer in the memory of the Us4R device
     *   (a single element of the buffer is an output of a single tx/rx sequence execution)
     * @param outputBuffer output buffer specification
     * @param workMode scheme work mode
     * @param digitalDownConversion DDC parameters
     */
    Scheme(TxRxSequence txRxSequence, uint16 rxBufferSize, const framework::DataBufferSpec &outputBuffer,
           WorkMode workMode, DigitalDownConversion digitalDownConversion)
        : txRxSequence(std::move(txRxSequence)), rxBufferSize(rxBufferSize), outputBuffer(outputBuffer),
          workMode(workMode), ddc(std::move(digitalDownConversion)) {}

    Scheme(TxRxSequence txRxSequence, uint16 rxBufferSize, const framework::DataBufferSpec &outputBuffer,
           WorkMode workMode, DigitalDownConversion ddc, const std::vector<arrus::framework::NdArray> &constants)
        : txRxSequence(txRxSequence), rxBufferSize(rxBufferSize), outputBuffer(outputBuffer), workMode(workMode),
          ddc(std::move(ddc)), constants(constants) {}

    Scheme(TxRxSequence txRxSequence, uint16 rxBufferSize, const framework::DataBufferSpec &outputBuffer,
           WorkMode workMode, const std::vector<arrus::framework::NdArray> &constants)
        : txRxSequence(txRxSequence), rxBufferSize(rxBufferSize), outputBuffer(outputBuffer), workMode(workMode),
          ddc(std::nullopt), constants(constants) {}

    const TxRxSequence &getTxRxSequence() const { return txRxSequence; }

    uint16 getRxBufferSize() const { return rxBufferSize; }

    const framework::DataBufferSpec &getOutputBuffer() const { return outputBuffer; }

    WorkMode getWorkMode() const { return workMode; }

    const std::optional<DigitalDownConversion> &getDigitalDownConversion() const { return ddc; }

    const std::vector<arrus::framework::NdArray> &getConstants() const { return constants; }

private:
    TxRxSequence txRxSequence;
    uint16 rxBufferSize;
    ::arrus::framework::DataBufferSpec outputBuffer;
    WorkMode workMode;
    std::optional<DigitalDownConversion> ddc;
    std::vector<arrus::framework::NdArray> constants;
};

}// namespace arrus::ops::us4r

#endif//ARRUS_CORE_API_OPS_US4R_H
