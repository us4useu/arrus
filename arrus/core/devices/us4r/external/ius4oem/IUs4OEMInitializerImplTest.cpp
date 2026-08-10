#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "arrus/core/devices/us4r/tests/MockIUs4OEM.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMInitializerImpl.h"

namespace {

using namespace arrus::devices;
using ::testing::Return;
using ::testing::InSequence;
using ::testing::Sequence;

// Test if the input array is approprietaly sorted
TEST(IUs4OEMInitializerImplTest, Us4OEMsSortedApproprietaly) {
    std::vector<IUs4OEMHandle> ius4oems;
    ius4oems.emplace_back(std::make_unique<::testing::NiceMock<MockIUs4OEM>>());
    ius4oems.emplace_back(std::make_unique<::testing::NiceMock<MockIUs4OEM>>());
    ius4oems.emplace_back(std::make_unique<::testing::NiceMock<MockIUs4OEM>>());
    ON_CALL(GET_MOCK_PTR(ius4oems[0]), getId)
        .WillByDefault(Return(4));
    ON_CALL(GET_MOCK_PTR(ius4oems[1]), getId)
            .WillByDefault(Return(0));
    ON_CALL(GET_MOCK_PTR(ius4oems[2]), getId)
            .WillByDefault(Return(2));

    IUs4OEMInitializerImpl initializer;
    initializer.sortModulesById(ius4oems);
    initializer.initModules(ius4oems);

    EXPECT_EQ(ius4oems[0]->getId(), 0);
    EXPECT_EQ(ius4oems[1]->getId(), 2);
    EXPECT_EQ(ius4oems[2]->getId(), 4);
}

// Test the order of initialization.

TEST(IUs4OEMInitializerImplTest, Us4OEMsInitializedProperly) {
    std::vector<IUs4OEMHandle> ius4oems;

    ius4oems.emplace_back(std::make_unique<::testing::NiceMock<MockIUs4OEM>>());
    ius4oems.emplace_back(std::make_unique<::testing::NiceMock<MockIUs4OEM>>());

    ON_CALL(GET_MOCK_PTR(ius4oems[0]), getId)
            .WillByDefault(Return(4));
    ON_CALL(GET_MOCK_PTR(ius4oems[1]), getId)
            .WillByDefault(Return(0));

    {
        Sequence us4oem0Seq;
        Sequence us4oem1Seq;

        EXPECT_CALL(GET_MOCK_PTR(ius4oems[1]), initialize(1)).InSequence(us4oem1Seq);
        EXPECT_CALL(GET_MOCK_PTR(ius4oems[0]), initialize(1)).InSequence(us4oem0Seq);

        EXPECT_CALL(GET_MOCK_PTR(ius4oems[1]), synchronize()).InSequence(us4oem1Seq, us4oem0Seq);
        EXPECT_CALL(GET_MOCK_PTR(ius4oems[1]), initialize(2)).InSequence(us4oem1Seq);
        EXPECT_CALL(GET_MOCK_PTR(ius4oems[0]), initialize(2)).InSequence(us4oem0Seq);

        EXPECT_CALL(GET_MOCK_PTR(ius4oems[1]), synchronize()).InSequence(us4oem1Seq, us4oem0Seq);
        EXPECT_CALL(GET_MOCK_PTR(ius4oems[1]), initialize(3)).InSequence(us4oem1Seq);
        EXPECT_CALL(GET_MOCK_PTR(ius4oems[0]), initialize(3)).InSequence(us4oem0Seq);

        EXPECT_CALL(GET_MOCK_PTR(ius4oems[1]), synchronize()).InSequence(us4oem1Seq, us4oem0Seq);
        EXPECT_CALL(GET_MOCK_PTR(ius4oems[1]), initialize(4)).InSequence(us4oem1Seq);
        EXPECT_CALL(GET_MOCK_PTR(ius4oems[0]), initialize(4)).InSequence(us4oem0Seq);
    }

    IUs4OEMInitializerImpl initializer;
    initializer.sortModulesById(ius4oems);
    initializer.initModules(ius4oems);
}


}


