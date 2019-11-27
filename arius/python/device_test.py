import unittest
from arius.python.test_tools import mock_import
import numpy as np

# Module mocks.
class AriusMock:
    pass

mock_import(
    "arius.python.iarius",
    Arius=AriusMock
)
# Project imports.
from arius.python.device import (
    Device,
    Probe,
    ProbeHardwareSubaperture,
    Subaperture
)


# Class mocks
class AriusCardMock(Device):
    def __init__(self, index, n_rx_channels, n_tx_channels, mock_data=None):
        super().__init__("AriusCardMock", index)
        self.n_tx_channels = n_tx_channels
        self.n_rx_channels = n_rx_channels
        self.tx_aperture = None
        self.tx_delays = [0.0]*self.n_tx_channels
        self.tx_frequency = None
        self.tx_periods = None
        self.rx_apertures = []
        self.rx_total_bytes = 0
        self.mock_data = mock_data

    def get_n_rx_channels(self):
        return self.n_rx_channels

    def get_n_tx_channels(self):
        return self.n_tx_channels

    def start_if_necessary(self):
        print("MockCard started.")

    def set_tx_aperture(self, origin, size):
        self.tx_aperture = Subaperture(origin, size)

    def set_tx_delays(self, delays):
        for i, delay in enumerate(delays):
            self.tx_delays[i] = delay

    def set_tx_frequency(self, f):
        self.tx_frequency = f

    def set_tx_periods(self, n):
        self.tx_periods = n

    def set_rx_aperture(self, origin, size):
        if self.mock_data is not None:
            self.rx_apertures.append(Subaperture(origin, size))

    def schedule_receive(self, address, length):
        if self.mock_data is not None:
            self.rx_total_bytes += length

    def sw_trigger(self):
        pass

    def wait_until_sgdma_finished(self):
        pass

    def transfer_rx_buffer_to_host(self, dst_array, src_addr):
        if self.mock_data is not None:
            last_subaperture = self.rx_apertures[-1]
            origin, size = last_subaperture.origin, last_subaperture.size
            dst_array[:, :] = self.mock_data[:, origin:(origin+size)]

    def set_rx_time(self, rx_time):
        self.rx_time = rx_time


class ProbeRxTest(unittest.TestCase):

    def test_probe_sets_rx_for_two_cards(self):
        # Set.
        hw_subapertures = [
            ProbeHardwareSubaperture(
                card=AriusCardMock(
                    0, n_rx_channels=32, n_tx_channels=128,
                    mock_data=np.tile(np.array(range(0, 128)), 4096).reshape((4096, 128))
                ),
                origin=0,
                size=128
            ),
            ProbeHardwareSubaperture(
                card=AriusCardMock(
                    1, n_rx_channels=32, n_tx_channels=128,
                    mock_data=np.tile(np.array(range(128, 192)), 4096).reshape((4096, 64))
                ),
                origin=0,
                size=64
            )
        ]

        # Run.
        probe = self._create_probe(hw_subapertures, 0)
        tx_aperture = Subaperture(0, 192)
        tx_delays = list(range(0, 192))
        carrier_frequency = 14e6
        n_periods = 1
        rf = probe.transmit_and_record(
            tx_aperture=tx_aperture,
            tx_delays=tx_delays,
            carrier_frequency=carrier_frequency,
            n_tx_periods=n_periods,
            n_samples=4096
        )
        # Verify.
        card0 = hw_subapertures[0].card
        card1 = hw_subapertures[1].card
        self.assertListEqual(
            [
                Subaperture(0, 32),
                Subaperture(32, 32),
                Subaperture(64, 32),
                Subaperture(96, 32)
            ],
            card0.rx_apertures)
        self.assertEqual(128*4096*probe.dtype.itemsize, card0.rx_total_bytes)
        self.assertListEqual(
            [
                Subaperture(0, 32),
                Subaperture(32, 32),
            ],
            card1.rx_apertures)
        self.assertEqual(64*4096*probe.dtype.itemsize, card1.rx_total_bytes)
        # First row of RF matrix contains expected pattern.
        self.assertTrue((rf[0, :] == list(range(0, 192))).all())
        # All rows are the same.
        self.assertTrue((rf[0, :] == rf).all())


    def _create_probe(self, apertures, master_card_idx):
        return Probe(
            index=0,
            model_name="test_probe",
            hw_subapertures=apertures,
            master_card=apertures[master_card_idx].card)


class ProbeTxTest(unittest.TestCase):
    def test_probe_sets_tx_for_single_card(self):
        # Set.
        hw_subapertures = [
            ProbeHardwareSubaperture(
                card=AriusCardMock(0, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=32 # Determines the size of the probe's aperture.
            )
        ]
        # Run.
        probe = self._create_probe(hw_subapertures, 0)
        tx_aperture = Subaperture(0, 32)
        tx_delays = list(range(0, 32))
        carrier_frequency = 5e6
        n_periods = 1
        probe.transmit_and_record(
            tx_aperture=tx_aperture,
            tx_delays=tx_delays,
            carrier_frequency=carrier_frequency,
            n_tx_periods=n_periods
        )
        # Verify.
        self.assertEqual(tx_aperture, hw_subapertures[0].card.tx_aperture)
        self.assertEqual(carrier_frequency, hw_subapertures[0].card.tx_frequency)
        self.assertEqual(n_periods, hw_subapertures[0].card.tx_periods)

        self._assert_card_delays(
            [
                tx_delays + [0.0]*96
            ],
            hw_subapertures
        )

    def test_probe_sets_tx_for_single_card_hw_offset(self):
        # Set.
        hw_subapertures = [
            ProbeHardwareSubaperture(
                card=AriusCardMock(0, n_rx_channels=32, n_tx_channels=128),
                origin=34,
                size=32
            )
        ]

        # Run.
        probe = self._create_probe(hw_subapertures, 0)
        tx_aperture = Subaperture(0, 16)
        tx_delays = list(range(0, 16))
        carrier_frequency = 5e6
        n_periods = 1
        probe.transmit_and_record(
            tx_aperture=tx_aperture,
            tx_delays=tx_delays,
            carrier_frequency=carrier_frequency,
            n_tx_periods=n_periods
        )
        # Verify.
        self.assertEqual(Subaperture(34, 16),
                         hw_subapertures[0].card.tx_aperture)
        self.assertEqual(carrier_frequency, hw_subapertures[0].card.tx_frequency)
        self.assertEqual(n_periods, hw_subapertures[0].card.tx_periods)
        print(tx_delays)
        print(probe.hw_subapertures[0].card.tx_delays)
        self._assert_card_delays(
            [
                34*[0.0] + tx_delays + [0.0]*(128-(34+len(tx_delays))),
            ],
            hw_subapertures
        )


    def test_probe_sets_tx_for_single_card_offset_origin(self):
        # Set.
        hw_subapertures = [
            ProbeHardwareSubaperture(
                card=AriusCardMock(0, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=64
            )
        ]

        # Run.
        probe = self._create_probe(hw_subapertures, 0)
        tx_aperture = Subaperture(14, 32)
        tx_delays = list(range(0, 32))
        carrier_frequency = 10e6
        n_periods = 2
        probe.transmit_and_record(
            tx_aperture=tx_aperture,
            tx_delays=tx_delays,
            carrier_frequency=carrier_frequency,
            n_tx_periods=n_periods
        )
        # Verify.
        self.assertEqual(tx_aperture, hw_subapertures[0].card.tx_aperture)
        self.assertEqual(carrier_frequency, hw_subapertures[0].card.tx_frequency)
        self.assertEqual(n_periods, hw_subapertures[0].card.tx_periods)
        self._assert_card_delays(
            [
                14*[0.0] + tx_delays + [0.0]*(128-(14+len(tx_delays))),
            ],
            hw_subapertures
        )

    def test_probe_sets_tx_for_two_cards(self):
        # Set.
        hw_subapertures = [
            ProbeHardwareSubaperture(
                card=AriusCardMock(0, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=128
            ),
            ProbeHardwareSubaperture(
                card=AriusCardMock(1, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=64
            )
        ]

        # Run.
        probe = self._create_probe(hw_subapertures, 0)
        tx_aperture = Subaperture(0, 192)
        tx_delays = list(range(0, 192))
        carrier_frequency = 14e6
        n_periods = 1
        probe.transmit_and_record(
            tx_aperture=tx_aperture,
            tx_delays=tx_delays,
            carrier_frequency=carrier_frequency,
            n_tx_periods=n_periods
        )
        # Verify.
        self.assertEqual(Subaperture(0, 128),
                         hw_subapertures[0].card.tx_aperture)
        self.assertEqual(Subaperture(0, 64),
                         hw_subapertures[1].card.tx_aperture)
        self._assert_card_delays(
            [
                list(range(0, 128)),
                list(range(128, 192)) + [0.0]*64
            ],
            hw_subapertures
        )
        for hws in hw_subapertures:
            self.assertEqual(carrier_frequency, hws.card.tx_frequency)
            self.assertEqual(n_periods, hws.card.tx_periods)

    def test_probe_sets_tx_for_two_cards_complete_apertures(self):
        # Set.
        hw_subapertures = [
            ProbeHardwareSubaperture(
                card=AriusCardMock(0, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=128
            ),
            ProbeHardwareSubaperture(
                card=AriusCardMock(1, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=128
            )
        ]

        # Run.
        probe = self._create_probe(hw_subapertures, 0)
        tx_aperture = Subaperture(0, 256)
        tx_delays = list(range(0, 256))
        carrier_frequency = 14e6
        n_periods = 1
        probe.transmit_and_record(
            tx_aperture=tx_aperture,
            tx_delays=tx_delays,
            carrier_frequency=carrier_frequency,
            n_tx_periods=n_periods
        )
        # Verify.
        self.assertEqual(Subaperture(0, 128),
                         hw_subapertures[0].card.tx_aperture)
        self.assertEqual(Subaperture(0, 128),
                         hw_subapertures[1].card.tx_aperture)
        self._assert_card_delays(
            [
                list(range(0, 128)),
                list(range(128, 256))
            ],
            hw_subapertures
        )
        for hws in hw_subapertures:
            self.assertEqual(carrier_frequency, hws.card.tx_frequency)
            self.assertEqual(n_periods, hws.card.tx_periods)


    def test_probe_sets_tx_for_two_cards_only_first_card_aperture(self):
        # Two cards, tx aperture only on the first card
        # Set.
        hw_subapertures = [
            ProbeHardwareSubaperture(
                card=AriusCardMock(0, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=128
            ),
            ProbeHardwareSubaperture(
                card=AriusCardMock(1, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=64
            )
        ]

        # Run.
        probe = self._create_probe(hw_subapertures, 0)
        tx_aperture = Subaperture(16, 64)
        tx_delays = list(range(0, 64))
        carrier_frequency = 8e6
        n_periods = 1

        probe.transmit_and_record(
            tx_aperture=tx_aperture,
            tx_delays=tx_delays,
            carrier_frequency=carrier_frequency,
            n_tx_periods=n_periods
        )
        # Verify.
        self.assertEqual(Subaperture(16, 64),
                         hw_subapertures[0].card.tx_aperture)
        self.assertIsNone(hw_subapertures[1].card.tx_aperture)
        self._assert_card_delays(
            [
                [0.0]*16 + list(range(0, 64)) + [0.0]*(128-(16+len(tx_delays))),
                [0.0]*128
            ],
            hw_subapertures
        )
        self.assertEqual(carrier_frequency, hw_subapertures[0].card.tx_frequency)
        self.assertEqual(n_periods, hw_subapertures[0].card.tx_periods)

    def test_probe_sets_tx_for_two_cards_only_second_card_aperture(self):
        # Two cards, tx aperture only on the second card

        # Set.
        hw_subapertures = [
            ProbeHardwareSubaperture(
                card=AriusCardMock(0, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=128
            ),
            ProbeHardwareSubaperture(
                card=AriusCardMock(1, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=64
            )
        ]

        # Run.
        probe = self._create_probe(hw_subapertures, 0)
        tx_aperture = Subaperture(130, 32)
        tx_delays = list(range(0, 32))
        carrier_frequency = 8e6
        n_periods = 1
        probe.transmit_and_record(
            tx_aperture=tx_aperture,
            tx_delays=tx_delays,
            carrier_frequency=carrier_frequency,
            n_tx_periods=n_periods
        )
        # Verify.
        self.assertEqual(Subaperture(2, 32),
                         hw_subapertures[1].card.tx_aperture)
        self.assertIsNone(hw_subapertures[0].card.tx_aperture)
        self._assert_card_delays(
            [
                [0.0]*128,
                [0.0]*2 + list(range(0, 32)) + [0.0]*(128-(2+len(tx_delays))),
            ],
            hw_subapertures
        )
        self.assertEqual(carrier_frequency, hw_subapertures[1].card.tx_frequency)
        self.assertEqual(n_periods, hw_subapertures[1].card.tx_periods)

    def test_probe_sets_tx_apeture_two_cards_second_hw_offset(self):
        # Two cards, second one's aperture starts at origin > 0
        # Set.
        hw_subapertures = [
            ProbeHardwareSubaperture(
                card=AriusCardMock(0, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=128
            ),
            ProbeHardwareSubaperture(
                card=AriusCardMock(1, n_rx_channels=32, n_tx_channels=128),
                origin=14,
                size=64
            )
        ]

        # Run.
        probe = self._create_probe(hw_subapertures, 0)
        tx_aperture = Subaperture(0, 192)
        tx_delays = list(range(0, 192))
        carrier_frequency = 8e6
        n_periods = 1
        probe.transmit_and_record(
            tx_aperture=tx_aperture,
            tx_delays=tx_delays,
            carrier_frequency=carrier_frequency,
            n_tx_periods=n_periods
        )
        # Verify.
        self.assertEqual(Subaperture(0, 128),
                         hw_subapertures[0].card.tx_aperture)
        self.assertEqual(Subaperture(14, 64),
                         hw_subapertures[1].card.tx_aperture)
        self._assert_card_delays(
            [
                list(range(0, 128)),
                [0.0]*14 + list(range(128, 192)) + [0.0]*50
            ],
            hw_subapertures
        )
        for hws in hw_subapertures:
            self.assertEqual(carrier_frequency, hws.card.tx_frequency)
            self.assertEqual(n_periods, hws.card.tx_periods)

    def test_single_element_aperture_card1(self):
        # Two cards, single element aperture near right border of the first card aperture
        # Set.
        hw_subapertures = [
            ProbeHardwareSubaperture(
                card=AriusCardMock(0, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=128
            ),
            ProbeHardwareSubaperture(
                card=AriusCardMock(1, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=64
            )
        ]

        # Run.
        probe = self._create_probe(hw_subapertures, 0)
        tx_aperture = Subaperture(127, 1)
        tx_delays = [1]
        carrier_frequency = 8e6
        n_periods = 1
        probe.transmit_and_record(
            tx_aperture=tx_aperture,
            tx_delays=tx_delays,
            carrier_frequency=carrier_frequency,
            n_tx_periods=n_periods
        )
        # Verify.
        self.assertEqual(Subaperture(127, 1),
                         hw_subapertures[0].card.tx_aperture)
        self.assertIsNone(hw_subapertures[1].card.tx_aperture)
        self._assert_card_delays(
            [
                [0.0]*127 + [1],
                [0.0]*128
            ],
            hw_subapertures
        )
        self.assertEqual(carrier_frequency, hw_subapertures[0].card.tx_frequency)
        self.assertEqual(n_periods, hw_subapertures[0].card.tx_periods)

    def test_single_element_aperture_card2(self):
        # Two cards, single element aperture near left bofder of the second card aperture
        # Set.
        hw_subapertures = [
            ProbeHardwareSubaperture(
                card=AriusCardMock(0, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=128
            ),
            ProbeHardwareSubaperture(
                card=AriusCardMock(1, n_rx_channels=32, n_tx_channels=128),
                origin=0,
                size=64
            )
        ]

        # Run.
        probe = self._create_probe(hw_subapertures, 0)
        tx_aperture = Subaperture(128, 1)
        tx_delays = [1]
        carrier_frequency = 8e6
        n_periods = 1
        probe.transmit_and_record(
            tx_aperture=tx_aperture,
            tx_delays=tx_delays,
            carrier_frequency=carrier_frequency,
            n_tx_periods=n_periods
        )
        # Verify.
        self.assertIsNone(hw_subapertures[0].card.tx_aperture)
        self.assertEqual(Subaperture(0, 1),
                         hw_subapertures[1].card.tx_aperture)
        self._assert_card_delays(
            [
                [0.0]*128,
                [1] + [0.0]*127
            ],
            hw_subapertures
        )
        self.assertEqual(carrier_frequency, hw_subapertures[1].card.tx_frequency)
        self.assertEqual(n_periods, hw_subapertures[1].card.tx_periods)


    def _create_probe(self, apertures, master_card_idx):
        return Probe(
            index=0,
            model_name="test_probe",
            hw_subapertures=apertures,
            master_card=apertures[master_card_idx].card)

    def _assert_card_delays(
            self,
            expected_delays,
            hw_subapertures
    ):
        for hws, expected_delay in zip(hw_subapertures, expected_delays):
            card = hws.card
            self.assertListEqual(expected_delay, card.tx_delays)


if __name__ == "__main__":
    unittest.main()
