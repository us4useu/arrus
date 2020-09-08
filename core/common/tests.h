#ifndef ARRUS_CORE_COMMON_TESTS_H
#define ARRUS_CORE_COMMON_TESTS_H

namespace arrus {

#define ARRUS_STRUCT_INIT_LIST(Type, initList)             \
    []() {                                      \
        Type x;                                 \
        (initList);                                  \
        return x;                               \
    }()
}

#endif //ARRUS_CORE_COMMON_TESTS_H
