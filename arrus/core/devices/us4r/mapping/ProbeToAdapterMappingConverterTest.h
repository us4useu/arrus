#ifndef ARRUS_ARRUS_CORE_DEVICES_US4R_MAPPING_PROBETOADAPTERMAPPINGCONVERTERTEST_H_
#define ARRUS_ARRUS_CORE_DEVICES_US4R_MAPPING_PROBETOADAPTERMAPPINGCONVERTERTEST_H_


#include <gtest/gtest.h>
#include <ostream>
#include "arrus/core/devices/probe/ProbeImpl.h"
#include "arrus/core/devices/us4r/FrameChannelMappingImpl.h"

namespace {

using namespace arrus;
using namespace arrus::devices;
using ::arrus::devices::FrameChannelMappingAddress;

struct GetPathRootTestCase {
    TxRxParametersSequence inputSequence;
    FrameChannelMapping::Handle inputFCM;
    TxRxParametersSequence outputSequence;
    FrameChannelMapping::Handle outputFCM;

    friend std::ostream &
    operator<<(std::ostream &os, const GetPathRootTestCase &aCase) {
        os << "path: " << aCase.path << " expectedRootTail: "
           << aCase.expectedRootTail.first
           << ", " << aCase.expectedRootTail.second;
        return os;
    }
};

class ProbeToAdapterMappingConverterTest: public ::testing::TestWithParam {
 protected:
    void SetUp() override {
    }
};

TEST_P(ProbeToadapterMappingConverterTest, OneToOne) {
    converter = ProbeToAdapterMappingConverter(const DeviceId &probeTxId, const DeviceId &probeRxId, ProbeSettings probeTx,
                    ProbeSettings probeRx, std::vector<ChannelIdx> txProbeMask,
                    std::vector<ChannelIdx> rxProbeMask, const ChannelIdx adapterNChannels)

}

TEST_F(ProbeToadapterMappingConverterTest, SingleAdapterSubaperture) {

}

TEST_F(ProbeToadapterMappingConverterTest, MultipleAdapterSubapertures) {

}

TEST_F(ProbeToadapterMappingConverterTest, NonStandard) {

}


}




#endif //ARRUS_ARRUS_CORE_DEVICES_US4R_MAPPING_PROBETOADAPTERMAPPINGCONVERTERTEST_H_
