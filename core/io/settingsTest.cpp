#include <gtest/gtest.h>
#include "arrus/core/api/io/settings.h"

TEST(ReatingProtoTxtFile, readsCorrectly) {
    SessionSettings settings = arrus::io::readSessionSettings("");
}
