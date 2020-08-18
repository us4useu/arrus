#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "arrus/core/devices/us4r/tests/MockIUs4OEM.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMFactory.h"
#include "arrus/core/devices/us4r/external/ius4oem/IUs4OEMInitializerImpl.h"

namespace {

using namespace arrus;
using ::testing::Return;
using ::testing::InSequence;

// Test if the input array is approprietaly sorted
TEST(IUs4OEMInitializerImplTest, Us4OEMsSortedApproprietaly) {
    std::vector<IUs4OEMHandle> ius4oems;
    ius4oems.emplace_back(std::make_unique<::testing::NiceMock<MockIUs4OEM>>());
    ius4oems.emplace_back(std::make_unique<::testing::NiceMock<MockIUs4OEM>>());
    ius4oems.emplace_back(std::make_unique<::testing::NiceMock<MockIUs4OEM>>());
    ON_CALL(GET_MOCK_PTR(ius4oems[0]), GetID)
        .WillByDefault(Return(4));
    ON_CALL(GET_MOCK_PTR(ius4oems[1]), GetID)
            .WillByDefault(Return(0));
    ON_CALL(GET_MOCK_PTR(ius4oems[2]), GetID)
            .WillByDefault(Return(2));

    IUs4OEMInitializerImpl initializer;
    initializer.initModules(ius4oems);

    EXPECT_EQ(ius4oems[0]->GetID(), 0);
    EXPECT_EQ(ius4oems[1]->GetID(), 2);
    EXPECT_EQ(ius4oems[2]->GetID(), 4);
}

// Test the order of initialization.

TEST(IUs4OEMInitializerImplTest, Us4OEMsInitializedProperly) {
    std::vector<IUs4OEMHandle> ius4oems;

    ius4oems.emplace_back(std::make_unique<::testing::NiceMock<MockIUs4OEM>>());
    ius4oems.emplace_back(std::make_unique<::testing::NiceMock<MockIUs4OEM>>());

    ON_CALL(GET_MOCK_PTR(ius4oems[0]), GetID)
            .WillByDefault(Return(4));
    ON_CALL(GET_MOCK_PTR(ius4oems[1]), GetID)
            .WillByDefault(Return(0));

    // The actual order: 1, 0
    {
        InSequence seq;

        EXPECT_CALL(GET_MOCK_PTR(ius4oems[1]), Initialize(1));
        EXPECT_CALL(GET_MOCK_PTR(ius4oems[0]), Initialize(1));

        EXPECT_CALL(GET_MOCK_PTR(ius4oems[1]), Synchronize());
        EXPECT_CALL(GET_MOCK_PTR(ius4oems[1]), Initialize(2));
        EXPECT_CALL(GET_MOCK_PTR(ius4oems[0]), Initialize(2));

        EXPECT_CALL(GET_MOCK_PTR(ius4oems[1]), Synchronize());
        EXPECT_CALL(GET_MOCK_PTR(ius4oems[1]), Initialize(3));
        EXPECT_CALL(GET_MOCK_PTR(ius4oems[0]), Initialize(3));

        EXPECT_CALL(GET_MOCK_PTR(ius4oems[1]), Synchronize());
        EXPECT_CALL(GET_MOCK_PTR(ius4oems[1]), Initialize(4));
        EXPECT_CALL(GET_MOCK_PTR(ius4oems[0]), Initialize(4));
    }

    IUs4OEMInitializerImpl initializer;
    initializer.initModules(ius4oems);
}


}


