#include <gtest/gtest.h>
#include <ostream>

#include "arrus/core/api/common/exceptions.h"

#include "IRQEvent.h"

using namespace ::arrus::devices;

TEST(IRQEventTest, CorrectCallOrder) {
    IRQEvent event;
    event.resetCounters();
    for(int i = 0; i < 10; ++i) {
        event.notifyOne();
        event.wait(1000);
    }
    // OK, test passed.
}

TEST(IRQEventTest, ToManyNotifiesGivesUnhandledIRQException) {
    IRQEvent event;
    event.resetCounters();
    event.notifyOne();
    event.notifyOne();
    EXPECT_THROW(event.wait(), arrus::IllegalStateException);
}

TEST(IRQEventTest, TimeoutException) {
    IRQEvent event;
    event.resetCounters();
    event.notifyOne();
    event.wait();
    EXPECT_THROW(event.wait(500), arrus::TimeoutException);
}