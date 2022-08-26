#ifndef ARRUS_CORE_EXAMPLES_AFEDEMODFIRS_H
#define ARRUS_CORE_EXAMPLES_AFEDEMODFIRS_H

//FIR coefficients valid for 65 MHz clk

const int16_t fir10M[32] = {    32700,  30949,  27632,  23096,  17802,  12264,  6981,   2383,
                                -1222,  -3674,  -4962,  -5209,  -4642,  -3541,  -2196,  -866,
                                252,    1040,   1461,   1545,   1368,   1031,   635,    263, 
                                -32,    -225,   -320,   -336,   -300,   -236,   -160,   -82 };

#endif //ARRUS_CORE_EXAMPLES_AFEDEMODFIRS_H