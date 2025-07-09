#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include "TxWaveformSoftStartConverter.h"
#include "arrus/core/api/ops/us4r/Pulse.h"
#include "arrus/common/format.h"

namespace {
using namespace ::arrus::devices;
using namespace ::arrus::ops::us4r;

// Verifies if the given waveforms are (almost) equal. By "almost" we mean that the
inline void expectWaveformsEqual(const Waveform &actual, const Waveform &expected) {
    EXPECT_EQ(actual.getNRepetitions(), expected.getNRepetitions());
    for(size_t i = 0; i < expected.getSegments().size(); ++i) {
        const auto &actualSegment = actual.getSegments().at(i);
        const auto &expectedSegment = expected.getSegments().at(i);
        EXPECT_EQ(actualSegment.getState(), expectedSegment.getState());
        EXPECT_EQ(actualSegment.getDuration().size(), expectedSegment.getDuration().size());
        for(size_t j = 0; j < expectedSegment.getDuration().size(); ++j) {
            const auto actualDuration = actualSegment.getDuration().at(j);
            const auto expectedDuration = expectedSegment.getDuration().at(j);
            EXPECT_FLOAT_EQ(actualDuration, expectedDuration);
        }
    }
}

struct DutyCycleTestParams {
    float dutyCycle;
    Waveform input;
    Waveform expected;
    uint32_t nCycles{1};

    friend std::ostream &operator<<(std::ostream &os, const DutyCycleTestParams &params) {
        os << "dutyCycle: " << params.dutyCycle << " input: " << params.input << " expected: " << params.expected;
        return os;
    }

};

class TxWaveformSoftStartDutyCycleTest : public ::testing::TestWithParam<DutyCycleTestParams> {};

/** Sets a single waveform with duty cycle < 100%. Verified the correctness of the Waveform */
TEST_P(TxWaveformSoftStartDutyCycleTest, CorrectlySetsDutyCycle) {
    const auto &input = GetParam().input;
    const auto &expected = GetParam().expected;
    const float dutyCycle = GetParam().dutyCycle;
    const auto nCycles = GetParam().nCycles;
    TxWaveformSoftStartConverter converter{nCycles, 1.0f, {dutyCycle, }, {1.0f}};
    const auto output = converter.convert(input);
    expectWaveformsEqual(output, expected);
}

INSTANTIATE_TEST_SUITE_P(
    CorrectlySetsDutyCycle, TxWaveformSoftStartDutyCycleTest,
    testing::Values(
        // 0
        // Does not affect TX amplitude/polarity
        // - level 1
        DutyCycleTestParams{
            0.5f,
            Pulse(1e6f, 1, false, 1).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.25f/1e6f, 0.25f/1e6f, 0.25f/1e6f, 0.25f/1e6f},
                        // states
                        {1, 0, -1, 0}
                    }
                },
                // repeats
                {1}
            }
        },
        // 1
        // - level 1, inversed
        DutyCycleTestParams{
            0.5f,
            Pulse(1e6f, 1, true, 1).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.25f/1e6f, 0.25f/1e6f, 0.25f/1e6f, 0.25f/1e6f},
                        // states
                        {-1, 0, 1, 0}
                    }
                },
                // repeats
                {1}
            }
        },
        // 2
        // - level 2
        DutyCycleTestParams{
            0.5f,
            Pulse(1e6f, 1, false, 2).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.25f/1e6f, 0.25f/1e6f, 0.25f/1e6f, 0.25f/1e6f},
                        // states
                        {2, 0, -2, 0}
                    }
                },
                // repeats
                {1}
            }
        },
        // 3
        // - level 2, inversed
        DutyCycleTestParams{
            0.5f,
            Pulse(1e6f, 1, true, 2).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.25f/1e6f, 0.25f/1e6f, 0.25f/1e6f, 0.25f/1e6f},
                        // states
                        {-2, 0, 2, 0}
                    }
                },
                // repeats
                {1}
            }
        },
        // 4
        // Does not affect the number of repetitions (NOTE: this test was set to apply the reduced duty cycle for longer TX)
        // even number of cycles
        DutyCycleTestParams{
            0.25f,
            Pulse(1e6f, 8, false, 2).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.125f/1e6f, 0.375f/1e6f, 0.125f/1e6f, 0.375f/1e6f},
                        // states
                        {2, 0, -2, 0}
                    }
                },
                // repeats
                {8}
            }
        },
        // 5
        // large number of even number of cycles
        DutyCycleTestParams{
            0.25f,
            Pulse(1e6f, 120, false, 2).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.125f/1e6f, 0.375f/1e6f, 0.125f/1e6f, 0.375f/1e6f},
                        // states
                        {2, 0, -2, 0}
                    }
                },
                // repeats
                {120}
            }
        },
        // 6
        // Odd number of cycles
        DutyCycleTestParams{
            0.25f,
            Pulse(1e6f, 7, false, 2).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.125f/1e6f, 0.375f/1e6f, 0.125f/1e6f, 0.375f/1e6f},
                        // states
                        {2, 0, -2, 0}
                    }
                },
                // repeats
                {7}
            }
        },
        // 7
        // Odd number of cycles
        DutyCycleTestParams{
            0.25f,
            Pulse(1e6f, 127, false, 2).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.125f/1e6f, 0.375f/1e6f, 0.125f/1e6f, 0.375f/1e6f},
                        // states
                        {2, 0, -2, 0}
                    }
                },
                // repeats
                {127}
            }
        },
        // 8
        // half period
        DutyCycleTestParams{
            0.25f,
            Pulse(1e6f, 2.5f, false, 2).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.125f/1e6f, 0.375f/1e6f, 0.125f/1e6f, 0.375f/1e6f},
                        // states
                        {2, 0, -2, 0}
                    },
                    // half-period
                    WaveformSegment {
                        // duration
                        {0.5e-6f},
                        // states
                        {2}
                    }
                },
                // repeats
                {2, 1}
            },
            2 // to make sure that the integer part will be soft-started, the half part will be intact
        },
        // Check different duty cycles
        // 9
        DutyCycleTestParams{
            0.1f,
            Pulse(1e6f, 1, false, 1).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.05f/1e6f, 0.45f/1e6f, 0.05f/1e6f, 0.45f/1e6f},
                        // states
                        {1, 0, -1, 0}
                    }
                },
                // repeats
                {1}
            }
        },
        // 10
        DutyCycleTestParams{
            0.9f,
            Pulse(1e6f, 1, false, 1).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.45f/1e6f, 0.05f/1e6f, 0.45f/1e6f, 0.05f/1e6f},
                        // states
                        {1, 0, -1, 0}
                    }
                },
                // repeats
                {1}
            }
        },
        // 11
        DutyCycleTestParams{
            0.2f,
            Pulse(1e6f, 1, false, 1).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.1f/1e6f, 0.4f/1e6f, 0.1f/1e6f, 0.4f/1e6f},
                        // states
                        {1, 0, -1, 0}
                    }
                },
                // repeats
                {1}
            }
        },
        // 12
        DutyCycleTestParams{
            0.25f,
            Pulse(1e6f, 1, false, 1).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.125f/1e6f, 0.375f/1e6f, 0.125f/1e6f, 0.375f/1e6f},
                        // states
                        {1, 0, -1, 0}
                    }
                },
                // repeats
                {1}
            }
        },
        // 13
        DutyCycleTestParams{
            0.75f,
            Pulse(1e6f, 1, false, 1).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.375f/1e6f, 0.125f/1e6f, 0.375f/1e6f, 0.125f/1e6f},
                        // states
                        {1, 0, -1, 0}
                    }
                },
                // repeats
                {1}
            }
        },
        // 14
        DutyCycleTestParams{
            1.0f,
            Pulse(1e6f, 1, false, 1).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.5f/1e6f, 0.5f/1e6f},
                        // states
                        {1, -1}
                    }
                },
                // repeats
                {1}
            }
        },
        // Does not affect center frequency
        // 15
        DutyCycleTestParams{
            0.75f,
            Pulse(10e6f, 1, false, 1).toWaveform(),
            Waveform {
                // segments
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.375f/10e6f, 0.125f/10e6f, 0.375f/10e6f, 0.125f/10e6f},
                        // states
                        {1, 0, -1, 0}
                    }
                },
                // repeats
                {1}
            }
        },
        // 15
        DutyCycleTestParams{
            0.75f,
            Pulse(4.2e6f, 1, false, 1).toWaveform(),
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.375f/4.2e6f, 0.125f/4.2e6f, 0.375f/4.2e6f, 0.125f/4.2e6f},
                        // states
                        {1, 0, -1, 0}
                    }
                },
                // repeats
                {1}
            }
        }
   )
);

// Check if the waveform is correctly divided into segments with varying duty cycles.
struct WaveformSplitTestParams {
    Pulse input;
    uint32_t nCycles;
    float ts;
    std::vector<float> dutyCycleDurationsFraction;
    std::vector<float> dutyCycles;
    Waveform expected;

    friend std::ostream &operator<<(std::ostream &os, const WaveformSplitTestParams &params) {
        os << "input pulse: " << params.input;
        os << " max n cycles: " << params.nCycles;
        os << " ts: " << params.ts;
        os << " dutyCycles: " << ::arrus::toString(params.dutyCycles);
        os << " dutyCycleDurationsFractions" << ::arrus::toString(params.dutyCycleDurationsFraction);
        os << " expected: " << params.expected;
        return os;
    }
};

class TxWaveformSoftStartWaveformSplitTest : public ::testing::TestWithParam<WaveformSplitTestParams> {};

/** Sets a single waveform with duty cycle < 100%. Verified the correctness of the Waveform */
TEST_P(TxWaveformSoftStartWaveformSplitTest, CorrectlySplitsWaveform) {
    const auto &input = GetParam().input;
    const auto &dutyCyclesDurationFractions = GetParam().dutyCycleDurationsFraction;
    const auto &dutyCycles = GetParam().dutyCycles;
    const auto ts = GetParam().ts;
    const auto nCycles = GetParam().nCycles;

    TxWaveformSoftStartConverter converter{nCycles, ts, dutyCycles, dutyCyclesDurationFractions};
    const auto output = converter.convert(input.toWaveform());
    const auto &expected = GetParam().expected;
    expectWaveformsEqual(output, expected);
}

INSTANTIATE_TEST_SUITE_P(
    CorrectlySplitsWaveform, TxWaveformSoftStartWaveformSplitTest,
    testing::Values(
        // 0
        // basic example
        WaveformSplitTestParams{
            Pulse(1e6f, 10, false, 1),
            1,
            6e-6f,
            {1.f/6.f, 2.f/6.f, 3.f/6.f},
            {0.25f, 0.5f, 0.75f},
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.125f/1e6f, 0.375f/1e6f, 0.125f/1e6f, 0.375f/1e6f},
                        // states
                        {1, 0, -1, 0}
                    },
                    WaveformSegment {
                        // duration
                        {0.25f/1e6f, 0.25f/1e6f, 0.25f/1e6f, 0.25f/1e6f},
                        // states
                        {1, 0, -1, 0}
                    },
                    WaveformSegment {
                        // duration
                        {0.375f/1e6f, 0.125f/1e6f, 0.375f/1e6f, 0.125f/1e6f},
                        // states
                        {1, 0, -1, 0}
                    },
                    WaveformSegment {
                        // duration
                        {0.5f/1e6f, 0.5f/1e6f},
                        // states
                        {1, -1}
                    }
                },
                // repeats
                {1, 2, 3, 4}
            }
        },
        // The pulse is shorter than the soft-start limit (pulse stops in between the 2nd and 3rd segment).
        WaveformSplitTestParams{
            // 4 cycles
            Pulse(1e6f, 3.0f, false, 1),
            1,
            // however, soft-start takes 6 cycles
            6e-6f,
            {1.f/6.f, 2.f/6.f, 3.f/6.f},
            {0.25f, 0.5f, 0.75f},
            Waveform {
                // segments
                {
                    WaveformSegment {
                        // duration
                        {0.125f/1e6f, 0.375f/1e6f, 0.125f/1e6f, 0.375f/1e6f},
                        // states
                        {1, 0, -1, 0}
                    },
                    WaveformSegment {
                        // duration
                        {0.25f/1e6f, 0.25f/1e6f, 0.25f/1e6f, 0.25f/1e6f},
                        // states
                        {1, 0, -1, 0}
                    },
                },
                // repeats
                {1, 2}
            }
        },
        // The pulse is longer than the soft-start limit only by 0.5 period -- the waveform should be ended with
        // a single state segment (half period).
        WaveformSplitTestParams{
            Pulse(1e6f, 6.5f, false, 1),
            5,
            6e-6f,
            {1.f/6.f, 2.f/6.f, 3.f/6.f},
            {0.25f, 0.5f, 0.75f},
            Waveform {
            // segments
                {
                    WaveformSegment {
                        // duration
                        {0.125f/1e6f, 0.375f/1e6f, 0.125f/1e6f, 0.375f/1e6f},
                        // states
                        {1, 0, -1, 0}
                    },
                    WaveformSegment {
                        // duration
                        {0.25f/1e6f, 0.25f/1e6f, 0.25f/1e6f, 0.25f/1e6f},
                        // states
                        {1, 0, -1, 0}
                    },
                    WaveformSegment {
                        // duration
                        {0.375f/1e6f, 0.125f/1e6f, 0.375f/1e6f, 0.125f/1e6f},
                        // states
                        {1, 0, -1, 0}
                    },
                    // Just a half-period
                    WaveformSegment {
                        // duration
                        {0.5f/1e6f},
                        // states
                        {1}
                    }
                },
                // repeats
                {1, 2, 3, 1}
            }
        }
    )
);

// TODO check the typical cases
// TODO Check converter accept

}
