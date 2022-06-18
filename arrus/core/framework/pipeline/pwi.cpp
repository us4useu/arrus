#include <algorithm>
#include <cmath>
#include <iostream>

#include "imaging/Metadata.h"
#include "imaging/ProbeModelExt.h"
#include "pwi.h"

using namespace ::arrus::session;
using namespace ::arrus::ops::us4r;
using namespace ::arrus::framework;

namespace arrus_example_imaging {

/**
 * A function that converts vector of delays with n elements
 * to a vector with m elements, where m is the the number
 * of (ARRUS) probe elements. The consecutive values delays
 * are applied on positions of consecutive 1s, 0s are applied
 * where the transmit elements are turned off.
 * Throws std::runtime_error when the number of active elements
 * is not equal to the length of input vector of delays.
 */
std::vector<float> toVectorOfDeviceDelays(const std::vector<bool> &txAperture, const std::vector<float> &delays) {
    std::vector<float> result(txAperture.size(), 0.0f);
    int j = 0;
    for(size_t i = 0; i < txAperture.size(); ++i) {
        if(txAperture[i]) {
            if(j >= delays.size()) {
                throw std::runtime_error("There is more active elements than delays.");
            }
            result[i] = delays[j];
            ++j;
        }
    }
    if(j < delays.size()) {
        throw std::runtime_error("There is more delays than active elements.");
    }
    return result;
}

/**
 * Note: this function works only with full TX/RX apertures.
 */
TxRxSequence createPwiSequence(const PwiSequence &seq, const std::vector<::arrus_example_imaging::ProbeModelExt> &probes) {
    // Apertures
    if (seq.getTxApertures().size() != seq.getRxApertures().size()
        || seq.getTxApertures().size() != seq.getAngles().size()) {
        throw std::runtime_error("There should be exactly the same number of tx, rx apertures and angles.");
    }
    auto nTxRxs = seq.getTxApertures().size();
    // Delays
    std::vector<float> centerDelays(nTxRxs);
    std::vector<::arrus_example_imaging::NdArray> delays(nTxRxs);

    float txCenterLateral = 0.0f;// Note: currently only a full TX aperture centered on lateral position is supported

    for (int i = 0; i < nTxRxs; ++i) {
        auto &txAperture = seq.getTxApertures()[i];
        auto &rxAperture = seq.getRxApertures()[i];
        auto angle = seq.getAngles()[i];

        auto &txProbe = probes[txAperture.getOrdinal()];
        const ::arrus_example_imaging::NdArray &positionLateral =
            txProbe.isOX() ? txProbe.getElementPositionX() : txProbe.getElementPositionY();
        const ::arrus_example_imaging::NdArray &positionAxial = txProbe.getElementPositionZ();

        ::arrus_example_imaging::NdArray d =
            (positionLateral * std::sin(angle) + positionAxial * std::cos(angle)) / seq.getSpeedOfSound();
        d = d - d.min<float>();
        delays[i] = d;
        auto nElements = d.getNumberOfElements();
        if (nElements % 2 == 0) {
            // Even number of elements in aperture: return average delay of the two center elements.
            centerDelays[i] = (d.get<float>(nElements / 2 - 1) + d.get<float>(nElements / 2)) / 2;
        } else {
            // Odd number of elements in aperture: return delay of the central element.
            centerDelays[i] = d.get<float>(nElements / 2);
        }
    }
    // Equalize the delay of the center of aperture.
    // The common delay applied for center of each TX aperture
    // So we can use a single TX center delay on image reconstruction step.
    // The center of transmit will be in the same position for all TX/RXs.
    float maxCenterDelay = *std::max_element(std::begin(centerDelays), std::end(centerDelays));

    for(int i = 0; i < nTxRxs; ++i) {
        delays[i] = delays[i]-(centerDelays[i]-maxCenterDelay);
    }

    std::vector<TxRx> txrxs;
    for (int i = 0; i < nTxRxs; ++i) {
        auto &txAperture = seq.getTxApertures()[i];
        auto &rxAperture = seq.getRxApertures()[i];
        // TODO note: ARRUS will require delays of active elements in future
        auto d = toVectorOfDeviceDelays(txAperture.getMask(), delays[i].toVector<float>());

        txrxs.emplace_back(Tx(txAperture.getMask(), d, seq.getPulse()),
                           Rx(rxAperture.getMask(), seq.getSampleRange(), seq.getDownsamplingFactor()), seq.getPri());
    }
    float sri = seq.getSri().has_value() ? seq.getSri().value() : TxRxSequence::NO_SRI;
    return TxRxSequence{txrxs, {}, sri};
}

/**
 * Uploads given TX/RX sequence on device.
 *
 * @param session session on which the sequence should be uploaded
 * @param seq sequence to upload
 * @param probes list of probe definitions used in this session, note:
 * @return a tuple: (output data buffer, output dimensions, output metadata)
 */
std::tuple<std::shared_ptr<::arrus::framework::Buffer>, NdArrayDef, std::shared_ptr<Metadata>>
upload(Session *session, const PwiSequence &seq, const std::vector<ProbeModelExt> &probes) {
    auto *us4r = (::arrus::devices::Us4R *) session->getDevice("/Us4R:0");

    auto txRxSequence = createPwiSequence(seq, probes);

    DataBufferSpec outputBuffer{DataBufferSpec::Type::FIFO, 4};
    Scheme scheme{txRxSequence, 2, outputBuffer, Scheme::WorkMode::HOST};
    auto result = session->upload(scheme);

    MetadataBuilder metadataBuilder;
    metadataBuilder.addObject(
        "frameChannelMapping",
        result.getConstMetadata()->get<::arrus::devices::FrameChannelMapping>("frameChannelMapping"));
    metadataBuilder.setValue("samplingFrequency", us4r->getSamplingFrequency() / seq.getDownsamplingFactor());
    metadataBuilder.addObject("sequence", std::make_shared<PwiSequence>(seq));
    metadataBuilder.addObject("rawSequence", std::make_shared<TxRxSequence>(txRxSequence));
    metadataBuilder.addObject("probeModels", std::make_shared<std::vector<ProbeModelExt>>(probes));

    // Determine output size.
    auto buffer = std::static_pointer_cast<DataBuffer>(result.getBuffer());
    if (buffer->getNumberOfElements() == 0) {
        throw std::runtime_error("The output buffer should have at least one element.");
    }
    auto outputShape = buffer->getElement(0)->getData().getShape().getValues();
    NdArrayDef outputDef{outputShape, DataType::INT16};
    return {result.getBuffer(), outputDef, metadataBuilder.buildSharedPtr()};
}
}
