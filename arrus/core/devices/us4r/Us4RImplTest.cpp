namespace {
TEST_F(ProbeAdapterChannelMappingEsaote3Test, CalculatesCorrectRxDelay) {
    std::vector<float> delays0(getNChannels(), 0.0f);
    std::vector<float> delays1(getNChannels(), 0.0f);
    BitMask txAperture(getNChannels(), true);
    BitMask rxAperture(getNChannels(), true);
    // Partially filled
    BitMask txAperture1(getNChannels(), false);
    std::fill(std::begin(txAperture1), std::begin(txAperture1) + 10, true);

    ops::us4r::Pulse pulse0{2.0e6f, 2.0f, false};
    ops::us4r::Pulse pulse1{3.0e6f, 3.0f, true};
    for (int i = 0; i < getNChannels(); ++i) {
        delays0[i] = i * 10e-7;
    }
    for (int i = 0; i < getNChannels(); ++i) {
        delays1[i] = 0.0f;
    }
    std::vector<TxRxParameters> seq = {// Linearly increasing TX delays.
                                       ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                                                              (x.txAperture = txAperture, x.rxAperture = rxAperture,
                                                               x.txDelays = delays0, x.pulse = pulse0))
                                           .getTxRxParameters(),
                                       // All TX delays the same.
                                       ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                                                              (x.txAperture = txAperture, x.rxAperture = rxAperture,
                                                               x.txDelays = delays1, x.pulse = pulse1))
                                           .getTxRxParameters(),
                                       // Partial TX aperture.
                                       ARRUS_STRUCT_INIT_LIST(TestTxRxParams,
                                                              (x.txAperture = txAperture1, x.rxAperture = rxAperture,
                                                               x.txDelays = delays0, x.pulse = pulse1))
                                           .getTxRxParameters()};

    float rxDelay0 = *std::max_element(std::begin(delays0), std::end(delays0))
        + 1.0f / pulse0.getCenterFrequency() * pulse0.getNPeriods();
    float rxDelay1 = *std::max_element(std::begin(delays1), std::end(delays1))
        + 1.0f / pulse1.getCenterFrequency() * pulse1.getNPeriods();
    float rxDelay2 = *std::max_element(std::begin(delays0), std::begin(delays0) + 10)
        + 1.0f / pulse1.getCenterFrequency() * pulse1.getNPeriods();

    EXPECT_SEQUENCE_PROPERTY_NFRAMES(
        0,
        ElementsAre(Property(&TxRxParameters::getRxDelay, rxDelay0), Property(&TxRxParameters::getRxDelay, rxDelay0),
                    Property(&TxRxParameters::getRxDelay, rxDelay0), Property(&TxRxParameters::getRxDelay, rxDelay1),
                    Property(&TxRxParameters::getRxDelay, rxDelay1), Property(&TxRxParameters::getRxDelay, rxDelay1),
                    Property(&TxRxParameters::getRxDelay, rxDelay2), Property(&TxRxParameters::getRxDelay, rxDelay2),
                    Property(&TxRxParameters::getRxDelay, rxDelay2)),
        9);
    EXPECT_SEQUENCE_PROPERTY_NFRAMES(
        1,
        ElementsAre(Property(&TxRxParameters::getRxDelay, rxDelay0), Property(&TxRxParameters::getRxDelay, rxDelay0),
                    Property(&TxRxParameters::getRxDelay, rxDelay0), Property(&TxRxParameters::getRxDelay, rxDelay1),
                    Property(&TxRxParameters::getRxDelay, rxDelay1), Property(&TxRxParameters::getRxDelay, rxDelay1),
                    Property(&TxRxParameters::getRxDelay, rxDelay2), Property(&TxRxParameters::getRxDelay, rxDelay2),
                    Property(&TxRxParameters::getRxDelay, rxDelay2)),
        9);
    SET_TX_RX_SEQUENCE(probeAdapter, seq);
}


}

