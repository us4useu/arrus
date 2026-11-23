#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "arrus/core/devices/us4r/validators/TxRxSequenceValidator.h"
#include "arrus/core/api/ops/us4r/TxRxSequence.h"

namespace {
using namespace arrus;
using namespace arrus::devices;
using namespace arrus::ops::us4r;

TxRx createTxRx(std::optional<Tgc> tgc, bool rxNop = false) {
    std::vector<bool> aperture(128, !rxNop);
    std::vector<float> delays(128, 0.0f);
    Tx tx{aperture, delays, Pulse{5e6f, 2.0f, false}};
    Rx rx{aperture, {0, 1024}, 1, {0, 0}, DeviceId(DeviceType::Probe, 0), std::move(tgc)};
    return TxRx{tx, rx, 100e-6f};
}

void validate(const TxRxSequence &seq) {
    TxRxSequenceValidator validator("test");
    validator.validate(seq);
    validator.throwOnErrors();
}

const Tgc CURVE{std::vector<float>{30.0f, 32.0f}};

TEST(TxRxSequenceValidatorTest, AcceptsNoTgc) {
    TxRxSequence seq{{createTxRx(std::nullopt), createTxRx(std::nullopt)}, {}};
    EXPECT_FALSE(seq.getTgc().has_value());
    EXPECT_NO_THROW(validate(seq));
}

TEST(TxRxSequenceValidatorTest, AcceptsTgcForSequenceOnly) {
    TxRxSequence seq{{createTxRx(std::nullopt), createTxRx(std::nullopt)}, std::vector<float>{30.0f, 31.0f}};
    EXPECT_TRUE(seq.getTgc().has_value());
    EXPECT_NO_THROW(validate(seq));
}

TEST(TxRxSequenceValidatorTest, AcceptsTgcForEachRx) {
    TxRxSequence seq{{createTxRx(CURVE), createTxRx(CURVE)}, {}};
    EXPECT_NO_THROW(validate(seq));
}

TEST(TxRxSequenceValidatorTest, AcceptsTgcForEachRxExceptNops) {
    TxRxSequence seq{{createTxRx(CURVE), createTxRx(std::nullopt, true)}, {}};
    EXPECT_NO_THROW(validate(seq));
}

TEST(TxRxSequenceValidatorTest, RejectsTgcForSequenceAndRx) {
    TxRxSequence seq{{createTxRx(CURVE), createTxRx(std::nullopt)}, std::vector<float>{30.0f}};
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST(TxRxSequenceValidatorTest, RejectsTgcForSomeRxOnly) {
    TxRxSequence seq{{createTxRx(CURVE), createTxRx(std::nullopt)}, {}};
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST(TxRxSequenceValidatorTest, RejectsEmptyRxTgc) {
    TxRxSequence seq{{createTxRx(Tgc{}), createTxRx(Tgc{})}, {}};
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST(TxRxSequenceValidatorTest, RejectsIncorrectNumberOfTimePoints) {
    Tgc tgc{std::vector<float>{0.0f, 1e-6f}, {30.0f}, std::string("a")};
    TxRxSequence seq{{createTxRx(std::nullopt)}, tgc};
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

TEST(TxRxSequenceValidatorTest, NamedEmptySequenceTgcIsConsideredSet) {
    TxRxSequence seq{{createTxRx(std::nullopt)}, Tgc{{}, std::string("a")}};
    EXPECT_TRUE(seq.getTgc().has_value());
    EXPECT_THROW(validate(seq), IllegalArgumentException);
}

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
