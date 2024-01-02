#ifndef ARRUS_CORE_DEVICES_US4R_MAPPING_INTERFACETOINTERFACEMAPPINGCONVERTER_H
#define ARRUS_CORE_DEVICES_US4R_MAPPING_INTERFACETOINTERFACEMAPPINGCONVERTER_H

#include "arrus/core/api/framework/NdArray.h"
#include "arrus/core/api/devices/us4r/FrameChannelMapping.h"


namespace arrus::devices::us4r {

using namespace ::arrus::framework;
using namespace ::arrus::devices;

class ProbeToAdapterMappingConverter {
public:
    /**
     * Probe <-> adapter (and adapter to probe) mapping converter.
     *
     * @param mappings probe to adapter mappings 1D array;, mappings[probe channel] = adapter channel
     * @param probeToBitstreamId probe to bistream ID, 1D array, probeToBitstreamId[probe ordinal] = bitstream id,
     *                           optional, nullopt means that the bitstream addresing is not used.
     */
    ProbeToAdapterMappingConverter(const std::vector<framework::NdArray> &mappings,
                                   const std::optional<framework::NdArray> &probeToBitstreamId):
    mappings(mappings), probeToAdapterBitstreamId(probeToBitstreamId) {}

    static framework::NdArray convert(const framework::NdArray &values, const framework::NdArray mapping) {

    }

    TxRxParametersSequence convert(const ops::us4r::TxRxSequence &sequence) {
        TxRxParametersSequenceBuilder sequenceBuilder;
        for(const auto& op: sequence.getOps()) {
            sequenceBuilder.addEntry(convert(op));
        }
        return std::move(sequenceBuilder.build());
    }

    TxRxParameters convert(const ops::us4r::TxRx &op) {
        TxRxParametersBuilder builder(op);
        auto txProbe = op.getTx().getPlacement().getOrdinal();
        auto rxProbe = op.getRx().getPlacement().getOrdinal();
        if(probeToAdapterBitstreamId.has_value()) {
            auto txBitstreamId = probeToAdapterBitstreamId[txProbe];
            auto rxBitstreamId = probeToAdapterBitstreamId[rxProbe];
            ARRUS_REQUIRES_EQUAL(txBitstreamId, rxBitstreamId,
                IllegalArgumentException(
                    format("Bitstream ids should be the same for RX and TX probes, got: {}(probe {}) and {}(probe {})",
                        rxProbe, op.getRx().getPlacement().toString(),
                        txProbe, op.getTx().getPlacement().toString()
                    )));
            // Determine bit stream ids for these probes.
            builder.setBitstreamId(txProbe);
        } else {
            builder.setBitstreamId(std::nullopt);
        }
        auto txAperture = convert(asarray(op.getTx().getAperture()), mappings.at(txProbe));
        auto txDelays = convert(asarray(op.getTx().getDelays()), mappings.at(txProbe));
        auto rxAperture = convert(asarray(op.getRx().getAperture()), mappings.at(rxProbe));

        builder.setTxAperture(txAperture.toVector<bool>());
        builder.setTxDelays(txDelays.toVector<float>());
        builder.setRxAperture(rxAperture.toVector<bool>());
        return builder.build();
    }

    FrameChannelMapping

private:
    std::vector<framework::NdArray> mappings;
    std::optional<framework::NdArray> probeToAdapterBitstreamId;
};
}// namespace arrus::devices::us4r

#endif// ARRUS_CORE_DEVICES_US4R_MAPPING_INTERFACETOINTERFACEMAPPINGCONVERTER_H