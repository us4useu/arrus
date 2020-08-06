#include <gtest/gtest.h>
#include "arrus/common/logging/Logger.h"
#include "arrus/common/logging/impl/Logging.h"
#include "arrus/core/common/logging.h"
#include "Us4OEMSettingsValidator.h"

TEST(LogTest, LogTest) {
    arrus::Us4OEMSettingsValidator validator;
    std::cerr << "Done!" << std::endl;
}

int main(int argc, char **argv) {
    INIT_ARRUS_TEST_LOG();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

