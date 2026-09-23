#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "Us4OEMTransferCallbacks.h"

namespace {

using namespace ::arrus::devices;

class Us4OEMTransferCallbacksTest : public ::testing::Test {
protected:
    /** Callbacks that record "<tag><element>.<transfer>". */
    Us4OEMTransferCallbacks::Callbacks createCallbacks(const std::string &tag, size_t nElements, size_t nPerElement) {
        Us4OEMTransferCallbacks::Callbacks result;
        for (size_t e = 0; e < nElements; ++e) {
            for (size_t t = 0; t < nPerElement; ++t) {
                result.emplace_back([this, tag, e, t]() {
                    calls.push_back(tag + std::to_string(e) + "." + std::to_string(t));
                });
            }
        }
        return result;
    }

    void interrupts(Us4OEMTransferCallbacks &callbacks, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            callbacks();
        }
    }

    std::vector<std::string> calls;
};

TEST_F(Us4OEMTransferCallbacksTest, CallsTheCallbacksInRoundRobinOrder) {
    Us4OEMTransferCallbacks callbacks;
    callbacks.set(createCallbacks("A", 2, 2), 2);
    interrupts(callbacks, 5);
    std::vector<std::string> expected = {"A0.0", "A0.1", "A1.0", "A1.1", "A0.0"};
    EXPECT_EQ(calls, expected);
}

TEST_F(Us4OEMTransferCallbacksTest, SwitchesToScheduledCallbacksAtTheGivenElement) {
    Us4OEMTransferCallbacks callbacks;
    callbacks.set(createCallbacks("A", 2, 2), 2);
    interrupts(callbacks, 2);// element 0
    // The elements 0 and 1 were triggered with the old callbacks, the element 2 is the first with the new ones.
    callbacks.schedule(createCallbacks("B", 2, 3), 3, 2);
    EXPECT_FALSE(callbacks.isScheduledInUse());
    interrupts(callbacks, 2 + 3 + 3);
    // element 2 -> the RX buffer element 0 (2 % 2), element 3 -> 1
    std::vector<std::string> expected = {"A0.0", "A0.1", "A1.0", "A1.1", "B0.0", "B0.1", "B0.2", "B1.0", "B1.1", "B1.2"};
    EXPECT_EQ(calls, expected);
    EXPECT_TRUE(callbacks.isScheduledInUse());
}

TEST_F(Us4OEMTransferCallbacksTest, KeepsTheRxBufferElementOrderAfterSwitching) {
    Us4OEMTransferCallbacks callbacks;
    callbacks.set(createCallbacks("A", 3, 1), 1);
    interrupts(callbacks, 2);// elements 0, 1
    // The element 2 (RX buffer element 2) is the first one acquired with the new callbacks.
    callbacks.schedule(createCallbacks("B", 3, 2), 2, 2);
    interrupts(callbacks, 4);
    std::vector<std::string> expected = {"A0.0", "A1.0", "B2.0", "B2.1", "B0.0", "B0.1"};
    EXPECT_EQ(calls, expected);
}

TEST_F(Us4OEMTransferCallbacksTest, DoesNotSwitchBeforeThePreviousElementsAreHandled) {
    Us4OEMTransferCallbacks callbacks;
    callbacks.set(createCallbacks("A", 2, 2), 2);
    // Elements 0 and 1 were triggered, but no interrupt has been handled yet.
    callbacks.schedule(createCallbacks("B", 2, 1), 1, 2);
    interrupts(callbacks, 5);
    std::vector<std::string> expected = {"A0.0", "A0.1", "A1.0", "A1.1", "B0.0"};
    EXPECT_EQ(calls, expected);
}

TEST_F(Us4OEMTransferCallbacksTest, ResetStartsFromTheFirstElement) {
    Us4OEMTransferCallbacks callbacks;
    callbacks.set(createCallbacks("A", 2, 1), 1);
    interrupts(callbacks, 1);
    callbacks.reset();
    interrupts(callbacks, 1);
    // Scheduled, but the device was restarted before switching: the new callbacks are used from the beginning.
    callbacks.schedule(createCallbacks("B", 2, 1), 1, 5);
    callbacks.reset();
    interrupts(callbacks, 1);
    std::vector<std::string> expected = {"A0.0", "A0.0", "B0.0"};
    EXPECT_EQ(calls, expected);
}

TEST_F(Us4OEMTransferCallbacksTest, QueuesMultipleScheduledCallbacks) {
    Us4OEMTransferCallbacks callbacks;
    callbacks.set(createCallbacks("A", 2, 1), 1);
    interrupts(callbacks, 1);// element 0
    // B is used from the element 2, C from the element 3 -- scheduled before B is in use.
    callbacks.schedule(createCallbacks("B", 2, 2), 2, 2);
    callbacks.schedule(createCallbacks("C", 2, 1), 1, 3);
    interrupts(callbacks, 1 + 2 + 1);
    std::vector<std::string> expected = {"A0.0", "A1.0", "B0.0", "B0.1", "C1.0"};
    EXPECT_EQ(calls, expected);
    EXPECT_TRUE(callbacks.isScheduledInUse());
    // The scheduled elements must not decrease.
    callbacks.schedule(createCallbacks("D", 2, 1), 1, 7);
    EXPECT_THROW(callbacks.schedule(createCallbacks("E", 2, 1), 1, 6), ::arrus::IllegalArgumentException);
}

TEST_F(Us4OEMTransferCallbacksTest, CountsTheCompletedElements) {
    Us4OEMTransferCallbacks callbacks;
    callbacks.set(createCallbacks("A", 2, 2), 2);
    EXPECT_EQ(callbacks.getNumberOfCompletedElements(), 0);
    interrupts(callbacks, 1);
    EXPECT_EQ(callbacks.getNumberOfCompletedElements(), 0);
    interrupts(callbacks, 1);
    EXPECT_EQ(callbacks.getNumberOfCompletedElements(), 1);
    // Switch at the element 2 (the element 1 is still handled by A).
    callbacks.schedule(createCallbacks("B", 2, 1), 1, 2);
    interrupts(callbacks, 2);
    EXPECT_EQ(callbacks.getNumberOfCompletedElements(), 2);
    EXPECT_FALSE(callbacks.isScheduledInUse());// B is adopted on its first interrupt
    interrupts(callbacks, 1);
    EXPECT_EQ(callbacks.getNumberOfCompletedElements(), 3);
    EXPECT_TRUE(callbacks.isScheduledInUse());
}

TEST_F(Us4OEMTransferCallbacksTest, RejectsInconsistentNumberOfCallbacks) {
    Us4OEMTransferCallbacks callbacks;
    EXPECT_THROW(callbacks.set(createCallbacks("A", 1, 3), 2), ::arrus::IllegalArgumentException);
    EXPECT_THROW(callbacks.schedule(createCallbacks("A", 1, 3), 0, 0), ::arrus::IllegalArgumentException);
}

}// namespace

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
