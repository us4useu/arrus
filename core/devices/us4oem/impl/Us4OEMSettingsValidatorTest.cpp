#include <gtest/gtest.h>

#include "Us4OEMSettingsValidator.h"
#include "core/common/logging/Logger.h"

TEST(LogTest, LogTest) {
    arrus::Logger logger(arrus::DeviceId(arrus::DeviceType::Us4OEM, 0));
    logger.log(arrus::LogSeverity::INFO, "Hello world");

    arrus::Logger::get().log(arrus::LogSeverity::INFO, "Hello world2");
}
