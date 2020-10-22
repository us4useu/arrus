#ifndef ARRUS_CORE_COMMON_TESTS_H
#define ARRUS_CORE_COMMON_TESTS_H

#include <iostream>

namespace arrus {

#define ARRUS_STRUCT_INIT_LIST(Type, initList)             \
    [&]() {                                      \
        Type x;                                 \
        (initList);                                  \
        return x;                               \
    }()

#define ARRUS_EXPECT_TENSORS_EQ(a, b) \
    do { \
        Eigen::Tensor<bool, 0> eq = ((a) == (b)).all(); \
        EXPECT_TRUE(eq(0)); \
    } while(0)

# define ARRUS_PRINT_DIFFERENCES(a, b) \
    do {                               \
        a.                                   \
    } while(0)
}

#endif //ARRUS_CORE_COMMON_TESTS_H
